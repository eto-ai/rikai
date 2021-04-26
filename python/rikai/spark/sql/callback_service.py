#  Copyright 2021 Rikai Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from py4j.java_gateway import CallbackServerParameters
from pyspark.sql import SparkSession

from rikai.logging import logger

__all__ = ["init_cb_service"]


def init_cb_service(spark: SparkSession):
    jvm = spark.sparkContext._gateway
    params = CallbackServerParameters(
        daemonize=True,
        daemonize_connections=True,
        # https://www.py4j.org/advanced_topics.html#using-py4j-without-pre-determined-ports-dynamic-port-number
        port=0,
        # Re-use the auth-token from the main java/spark process
        auth_token=jvm.gateway_parameters.auth_token,
    )

    jvm.start_callback_server(callback_server_parameters=params)

    python_port = jvm.get_callback_server().get_listening_port()
    jvm.java_gateway_server.resetCallbackClient(
        jvm.java_gateway_server.getCallbackClient().getAddress(),
        python_port,
    )
    logger.info("Spark callback server started")

    cb = CallbackService(spark)
    cb.register()


class CallbackService:
    """:py:class:`CallbackService` allows SparkSessions' JVM to run
    arbitrary code in SparkSession's python interpreter.

    Notes
    -----
    Internal use only
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.registry_map = {}

    def __repr__(self):
        return "PythonCbService"

    def resolve(self, registry_class: str, spec):
        """Resolve a ML model.

        Parameters
        ----------
        registry_class : str
            The full class name for the registry.
        uri : str
            The model URI.
        name : str
            Mode name
        options : dict[str, str]
            Options passed to the model.
        """
        if registry_class not in self.registry_map:
            from rikai.internal.reflection import find_class
            from rikai.spark.sql.codegen.base import Registry

            cls = find_class(registry_class)
            if not issubclass(cls, Registry):
                raise ValueError(
                    f"Class '{registry_class}' is not a Registry'"
                )
            self.registry_map[registry_class] = cls(self.spark)

        registry = self.registry_map[registry_class]
        try:
            return registry.resolve(spec)
        except Exception as e:
            raise RuntimeError(
                f"could not resolve {spec} from {registry}"
            ) from e

    def register(self):
        """Register this :py:class:`CallbackService` to SparkSession's JVM."""
        jvm = self.spark.sparkContext._jvm
        jvm.ai.eto.rikai.sql.spark.Python.register(self)
        logger.info(
            "Rikai Python CallbackService is registered to SparkSession"
        )

    def toString(self):
        """For Java compatibility"""
        return repr(self)

    class Java:
        implements = ["ai.eto.rikai.sql.spark.Python"]
