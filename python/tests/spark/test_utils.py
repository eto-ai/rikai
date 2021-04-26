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
import cv2
import numpy as np
from pathlib import Path

from pyspark.sql import DataFrame, Row, SparkSession

from rikai.spark.utils import df_to_rikai, Deduper
from rikai.testing.asserters import assert_count_equal
from rikai.types import Box2d, Image


def test_df_to_rikai(spark: SparkSession, tmp_path: Path):
    df = spark.createDataFrame(
        [Row(Box2d(1, 2, 3, 4)), Row(Box2d(23, 33, 44, 88))], ["bbox"]
    )
    df_to_rikai(df, str(tmp_path))
    actual_df = spark.read.format("rikai").load(str(tmp_path))
    assert_count_equal(df.collect(), actual_df.collect())


def test_Deduper(spark: SparkSession, asset_path: Path):
    image_uri = str(asset_path / "test_image.jpg")
    empty_array = np.zeros((300, 300, 3))
    _, empty_image = cv2.imencode(".jpg", empty_array)
    empty_image = empty_image.tostring()
    df = spark.createDataFrame(
        [
            Row(Image(image_uri)),
            Row(Image(image_uri)),
            Row(Image(empty_image)),
        ],
        ["images"],
    )
    deduper = Deduper(inputCol="images", outputCol="uid")
    df = deduper.transform(df)
    id_list = df.select("uid").rdd.flatMap(lambda x: x).collect()
    assert id_list[0] == id_list[1] != id_list[2]
