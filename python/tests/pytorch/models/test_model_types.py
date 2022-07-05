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

from pathlib import Path
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
import pytest
import torchvision


def _check_object_detection_models(spark: SparkSession, models: List[str]):
    uri = "https://i.scdn.co/image/ab67616d0000b273466def3ce70d94dcacb13c8d"
    for model in models:
        spark.sql(
            f"""CREATE OR REPLACE MODEL {model}
            FLAVOR pytorch
            MODEL_TYPE {model}
            """
        )
        df = spark.sql(
            f"select explode(ML_PREDICT({model}, to_image('{uri}')))"
        )
        assert df.count() >= 2


def test_ssd_models(spark: SparkSession):
    ssd_models = ["ssd", "ssdlite"]
    _check_object_detection_models(spark, ssd_models)


def test_fasterrcnn_models(spark: SparkSession):
    fasterrcnn_models = [
        "fasterrcnn",
        "fasterrcnn_resnet50_fpn",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "fasterrcnn_mobilenet_v3_large_320_fpn",
    ]
    _check_object_detection_models(spark, fasterrcnn_models)


def test_maskrcnn_models(spark: SparkSession):
    maskrcnn_models = ["maskrcnn"]
    _check_object_detection_models(spark, maskrcnn_models)


def test_retinanet_models(spark: SparkSession):
    retinanet_models = ["retinanet"]
    _check_object_detection_models(spark, retinanet_models)


def test_keypointrcnn_models(spark: SparkSession):
    keypointrcnn_models = ["keypointrcnn"]
    _check_object_detection_models(spark, keypointrcnn_models)


def _check_classification_models(
    spark: SparkSession, asset_path: Path, models: List[str]
):
    uri = str(asset_path / "cat.jpg")
    for model_name in models:
        spark.sql(
            f"""CREATE OR REPLACE MODEL {model_name}
            FLAVOR pytorch MODEL_TYPE {model_name}"""
        )
        df = spark.sql(f"SELECT ML_PREDICT({model_name}, to_image('{uri}'))")
        assert df.count() > 0
        assert df.schema == StructType(
            [
                StructField(
                    model_name,
                    StructType(
                        [
                            StructField("label_id", IntegerType()),
                            StructField("score", FloatType()),
                            StructField("label", StringType()),
                        ]
                    ),
                )
            ]
        )
        # Label(281) == "tabby, tabby cat"
        # Label(282) == "tiger cat"
        # https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
        result = df.first().asDict()[model_name]
        assert result.label_id in [281, 282]
        assert result.label in ["tabby", "tiger cat"]


def test_resnet(spark: SparkSession, asset_path: Path):
    models = [f"resnet{layers}" for layers in [18, 34, 50, 101, 152]]
    _check_classification_models(spark, asset_path, models)


def test_efficientnet(spark: SparkSession, asset_path: Path):
    if torchvision.__version__ < "0.11.0":
        pytest.skip("torchvision >= 0.11.0 is required")
    models = [f"efficientnet_b{scale}" for scale in range(8)]
    _check_classification_models(spark, asset_path, models)
