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

from pyspark.sql import SparkSession

from rikai.logging import logger

__all__ = ["CallbackService"]


class CallbackService(object):
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
        opt = model.javaOptions()
        options = {key: opt[key] for key in opt}
        py_class = model.pyClass()
        print(model.toString(), options, py_class)

    def register(self):
        jvm = self.spark.sparkContext._jvm
        jvm.ai.eto.rikai.sql.spark.Python.register(self)
        logger.info(
            "Rikai Python CallbackService is registered to SparkSession"
        )

    def toString(self):
        return repr(self)

    class Java:
        implements = ["ai.eto.rikai.sql.spark.Python"]
