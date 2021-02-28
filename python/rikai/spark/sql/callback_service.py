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
        # Re-use the auth-token from the main java/spark process
        auth_token=jvm.gateway_parameters.auth_token,
    )
    jvm.start_callback_server(callback_server_parameters=params)
    logger.info("Spark callback server started")

    cb = CallbackService(spark)
    cb.register()
    logger.info("Rikai Python callback service registered")


class CallbackService:
    """:py:class:`CallbackService` allows SparkSessions' JVM to run
    arbitrary code in SparkSession's python interpreter.

    Notes
    -----
    Internal use only
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def __repr__(self):
        return "PythonCbService"

    def codegen(self, model, temporary: bool):
        """Code generation for a Model.

        Parameters
        ----------
        model : jvm Model class
            The model to generate python code.
        temporary : bool
            Set true of the generated code will only be used once. Temporary
            code generation will be used when we use `ML_PREDICT` directly
            with a model URI.
        """
        opt = model.javaOptions()
        options = {key: opt[key] for key in opt}
        py_class = model.pyClass()
        logger.debug(f"Code generation for model={model.toString()}, options={options}"
                     f", pyClass=${py_class}, temporary=${temporary}")

    def register(self):
        """Register this :py:class:`CallbackService` to SparkSession's JVM."""
        jvm = self.spark.sparkContext._jvm
        jvm.ai.eto.rikai.sql.spark.Python.register(self)
        logger.info(
            "Rikai Python CallbackService is registered to SparkSession"
        )

    def toString(self):
        return repr(self)

    class Java:
        implements = ["ai.eto.rikai.sql.spark.Python"]
