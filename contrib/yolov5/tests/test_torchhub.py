#  Copyright (c) 2021 Rikai Authors
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

from pyspark.sql import Row, SparkSession

from rikai.contrib.yolov5.transforms import OUTPUT_SCHEMA
from rikai.types import Image


def test_yolov5(spark: SparkSession):
    work_dir = Path().absolute().parent.parent
    image_path = f"{work_dir}/python/tests/assets/test_image.jpg"
    spark.createDataFrame(
        [Row(image=Image(image_path))]
    ).createOrReplaceTempView("images")
    spark.sql(
        f"""
CREATE MODEL yolov5
FLAVOR yolov5
PREPROCESSOR 'rikai.contrib.yolov5.transforms.pre_processing'
POSTPROCESSOR 'rikai.contrib.yolov5.transforms.post_processing'
OPTIONS (device="cpu", batch_size=32)
RETURNS {OUTPUT_SCHEMA}
USING "torchhub:///ultralytics/yolov5:v6.0/yolov5s";
    """
    )
    result = spark.sql(
        f"""
    select ML_PREDICT(yolov5, image) as pred FROM images
    """
    )

    assert len(result.first().pred.boxes) > 0
