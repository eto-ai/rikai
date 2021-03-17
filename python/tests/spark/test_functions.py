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
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import col, lit, concat

# Rikai
from rikai.numpy import wrap
from rikai.spark.functions import (
    area,
    box2d_from_center,
    box2d,
    image_copy,
    numpy_to_image,
    video_to_images,
    spectrogram_image,
)
from rikai.types import Box2d, Image, VideoStream, YouTubeVideo


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


def test_numpy_to_image(spark: SparkSession, tmp_path: Path):
    """Test upload a numpy image to the external storage,
    and convert the data into Image asset.

    """
    df = spark.createDataFrame(
        [Row(id=1, data=wrap(np.ones((32, 32), dtype=np.uint8)))]
    )
    df = df.withColumn(
        "image",
        numpy_to_image(
            df.data, concat(lit(str(tmp_path)), lit("/"), df.id, lit(".png"))
        ),
    )
    df.count()
    # print(df.first().image)
    assert Path(df.first().image.uri) == tmp_path / "1.png"
    assert (tmp_path / "1.png").exists()


def test_video_to_images(spark: SparkSession):
    """Test extract video frames from YouTubeVideo/VideoStream types
    into list of Image assets.
    """
    sample_rate = 2
    start_frame = 0
    max_samples = 10
    videostream_df = spark.createDataFrame(
        [
            (
                VideoStream(
                    uri=os.path.abspath(
                        os.path.join(
                            os.path.dirname(__file__),
                            "..",
                            "assets",
                            "big_buck_bunny_short.mp4",
                        )
                    )
                ),
            ),
        ],
        ["video"],
    )
    youtube_df = spark.createDataFrame(
        [
            (YouTubeVideo(vid="rUWxSEwctFU"),),
        ],
        ["video"],
    )
    videostream_df = videostream_df.withColumn(
        "images",
        video_to_images(
            col("video"), lit(sample_rate), lit(start_frame), lit(max_samples)
        ),
    )
    youtube_df = youtube_df.withColumn(
        "images",
        video_to_images(
            col("video"), lit(sample_rate), lit(start_frame), lit(max_samples)
        ),
    )

    videostream_sample = videostream_df.first()["images"]
    youtube_sample = youtube_df.first()["images"]

    assert (
        type(videostream_sample) == list
        and type(videostream_sample[0]) == Image
        and len(videostream_sample) == max_samples
    )
    assert (
        type(youtube_sample) == list
        and type(youtube_sample[0]) == Image
        and len(youtube_sample) == max_samples
    )


def test_spectrogram_image(spark: SparkSession):
    """Test generate spectrogram image
    from YouTubeVideo/VideoStream videos types."""
    videostream_df = spark.createDataFrame(
        [
            (
                VideoStream(
                    uri=os.path.abspath(
                        os.path.join(
                            os.path.dirname(__file__),
                            "..",
                            "assets",
                            "big_buck_bunny_short.mp4",
                        )
                    )
                ),
            ),
        ],
        ["video"],
    )
    youtube_df = spark.createDataFrame(
        [
            (YouTubeVideo(vid="rUWxSEwctFU"),),
        ],
        ["video"],
    )
    videostream_df = videostream_df.withColumn(
        "spectrogram",
        spectrogram_image(col("video")),
    )
    youtube_df = youtube_df.withColumn(
        "spectrogram",
        spectrogram_image(col("video")),
    )
    videostream_sample = videostream_df.first()["spectrogram"]
    youtube_sample = youtube_df.first()["spectrogram"]

    assert type(videostream_sample) == Image
    assert type(youtube_sample) == Image
