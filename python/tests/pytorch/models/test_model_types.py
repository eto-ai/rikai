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

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

from rikai.spark.functions import init


def test_fasterrcnn_models(spark: SparkSession):
    init(spark)
    name = "fasterrcnn"
    spark.sql(
        f"CREATE MODEL {name} FLAVOR pytorch MODEL_TYPE fasterrcnn_mobilenet_v3_large_fpn"
    )
    uri = "https://i.scdn.co/image/ab67616d0000b273466def3ce70d94dcacb13c8d"
    df = spark.sql(
        f"select explode(ML_PREDICT(fasterrcnn, to_image('{uri}')))"
    )
    assert df.count() >= 3


def test_resnet(spark: SparkSession, asset_path: Path):
    init(spark)

    uri = str(asset_path / "cat.jpg")
    for layers in [18, 34, 50, 101, 152]:
        model_name = f"resnet{layers}"
        spark.sql(
            f"CREATE MODEL {model_name} FLAVOR pytorch MODEL_TYPE {model_name}"
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
                        ]
                    ),
                )
            ]
        )
        # Label(282) == "tiger cat"
        # https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
        assert df.first().asDict()[model_name].label_id == 282
