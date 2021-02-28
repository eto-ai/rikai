#  Copyright 2020 Rikai Authors
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
from rikai.spark.sql.callback_service import CallbackService

__all__ = ["RikaiSession"]


class RikaiSession:
    """Rikai session maintains the connection and callback server from Spark
    JVM."""

    def __init__(self, spark: SparkSession):
        assert spark is not None
        self.spark = spark
        self.callback_service = CallbackService(spark)
        self.started = False

    def __enter__(self):
        if not self.started:
            self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.started:
            self.stop()

    def start(self):
        jvm = self.spark.sparkContext._gateway
        params = CallbackServerParameters(
            daemonize=True,
            daemonize_connections=True,
            auth_token=jvm.gateway_parameters.auth_token,
        )
        jvm.start_callback_server(callback_server_parameters=params)
        logger.info("Spark callback server started")

        self.callback_service.register()
        logger.info("Rikai Python callback service registered")
        self.started = True

    def stop(self):
        # self.spark.sparkContext._gateway.shutdown_callback_server()
        logger.info("Spark callback server stopped")
