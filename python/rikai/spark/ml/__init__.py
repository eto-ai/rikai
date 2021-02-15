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

from rikai.spark.ml.model import ModelLoader


def _init(spark: SparkSession):
    """

    Parameters
    ----------
    spark : SparkSession

    Returns
    -------

    Warnings
    --------
    Internally use only

    """
    assert spark is not None
    jvm = spark.sparkContext._jvm
    spark.sparkContext._gateway.start_callback_server()

    # TODO: use jvm to register a callback
    jvm.ai.eto.rikai.sql.ModelLoaderRegistry.register(ModelLoader(spark))
