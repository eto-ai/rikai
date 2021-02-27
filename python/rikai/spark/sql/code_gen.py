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

from typing import Dict

from pyspark.sql import SparkSession

from rikai.logging import logger

__all__ = ["ModelCodeGen"]


class ModelCodeGen(object):
    """ModelCodeGen generate python code for a Model.

    Notes
    -----
    Internal use only
    """
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def __repr__(self):
        return "ModelResolver"

    def generate(self, model):
        print(model.toString())
        print(model.options)

    def register(self):
        jvm = self.spark.sparkContext._jvm
        jvm.ai.eto.rikai.sql.spark.ModelCodeGen.register(self)
        logger.info("Rikai ModelCodeGen(py) is registered to SparkSession")

    def toString(self):
        return repr(self)

    class Java:
        implements = ["ai.eto.rikai.sql.spark.ModelCodeGen"]
