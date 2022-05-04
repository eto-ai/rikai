#  Copyright 2022 Rikai Authors
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

import json
from pathlib import Path
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from rikai.parquet.dataset import Dataset
from rikai.spark.types import *
from rikai.types import *
from rikai.parquet.writer import df_to_rikai


def test_roundtrip(spark: SparkSession, tmp_path: Path):
    schema = StructType(
        fields=[
            StructField("image_id", StringType()),
            StructField("image", ImageType()),
            StructField(
                "annotations",
                ArrayType(
                    elementType=StructType(
                        fields=[
                            StructField("label", StringType()),
                            StructField("box", Box2dType()),
                        ]
                    )
                ),
            ),
        ]
    )
    df = pd.DataFrame(
        [
            {
                "image_id": "1",
                "image": Image("s3://bucket/path.jpg"),
                "annotations": [
                    {
                        "label": "car",
                        "box": Box2d(xmin=0, ymin=0, xmax=100, ymax=100),
                    }
                ],
            }
        ]
    )
    df_to_rikai(df, str(tmp_path), schema)

    pandas_df = pd.DataFrame(Dataset(str(tmp_path)))
    assert isinstance(pandas_df.image[0], Image)
    spark_df = spark.read.format("rikai").load(str(tmp_path))
    assert schema.json() == spark_df.schema.json()
