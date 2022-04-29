#  Copyright (c) 2022 Rikai Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import py4j
import pytest
from pyspark.sql import SparkSession
from utils import check_ml_predict

from rikai.spark.functions import init


def test_fasterrcnn_resnet50_fpn(spark: SparkSession, two_flickr_rows: list):
    name = "frf_model"
    spark.sql(
        f"""CREATE MODEL {name}
            FLAVOR pytorch
            MODEL_TYPE fasterrcnn_resnet50_fpn"""
    )
    check_ml_predict(spark, name, two_flickr_rows)


def test_ssdlite_with_option(spark: SparkSession):
    uri = "https://i.scdn.co/image/ab67616d0000b273466def3ce70d94dcacb13c8d"
    model = "ssdlite"
    spark.sql(
        f"""CREATE OR REPLACE MODEL {model}
        FLAVOR pytorch
        MODEL_TYPE {model}
        OPTIONS (min_score=1.0)
        """
    )
    df = spark.sql(f"select explode(ML_PREDICT({model}, to_image('{uri}')))")
    assert df.count() == 0
