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

"""Unit tests for Rikai provided spark UDFs."""

import os

import pandas as pd
import pandas.testing as pdt
from pyspark.sql.functions import col, lit
from pyspark.sql import Row, SparkSession

from rikai.spark.functions import area, image_copy, box2d, box2d_from_center
from rikai.types import Box2d, Image


def assert_area_equals(array, df):
    pdt.assert_frame_equal(
        pd.DataFrame({"area": array}),
        df.select("area").toPandas(),
        check_dtype=False,
    )


def test_areas(spark: SparkSession):
    """Test calculating bounding box's area."""
    df = spark.createDataFrame(
        [
            (Box2d(1, 2, 2.0, 3.0),),
            (Box2d(10, 12, 11.0, 17.0),),
        ],
        ["bbox"],
    )
    df = df.withColumn("area", area(col("bbox")))
    assert_area_equals([1.0, 5.0], df)


def test_box2d_udfs(spark):
    df = spark.createDataFrame(
        [
            Row(values=[1.0, 2.0, 2.0, 3.0]),
            Row(values=[10.0, 12.0, 11.0, 17.0]),
        ],
        ["values"],
    ).withColumn("bbox", box2d("values"))
    df = df.withColumn("area", area(col("bbox")))
    assert_area_equals([1.0, 5.0], df)


def test_box2d_center(spark):
    df = spark.createDataFrame(
        [
            Row(values=[1.5, 2.5, 1.0, 1.0]),
            Row(values=[10.5, 14.5, 1.0, 5.0]),
        ],
        ["values"],
    ).withColumn("bbox", box2d_from_center("values"))
    df = df.withColumn("area", area(col("bbox")))
    assert_area_equals([1.0, 5.0], df)


def test_box2d_top_left(spark: SparkSession):
    df = spark.createDataFrame(
        [
            Row(values=[1.0, 2.0, 1.0, 1.0]),
            Row(values=[10.0, 12.0, 1.0, 5.0]),
        ],
        ["values"],
    ).withColumn("bbox", box2d_from_center("values"))
    df = df.withColumn("area", area(col("bbox")))
    assert_area_equals([1.0, 5.0], df)


def test_image_copy(spark: SparkSession, tmpdir):
    source_image = os.path.join(tmpdir, "source_image")
    with open(source_image, "w") as fobj:
        fobj.write("abc")
    os.makedirs(os.path.join(tmpdir, "out"))

    df = spark.createDataFrame(
        [(Image(source_image),)], ["image"]
    )  # type: pyspark.sql.DataFrame
    df = df.withColumn(
        "image",
        image_copy(col("image"), lit(os.path.join(tmpdir, "out/"))),
    )
    data = df.collect()  # force lazy calculation
    out_file = os.path.join(tmpdir, "out", "source_image")
    assert Image(out_file) == data[0].image

    with open(os.path.join(out_file)) as fobj:
        assert fobj.read() == "abc"
