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

"""Domain-specific Pyspark UDFs
"""

from pyspark.sql import SparkSession

from rikai.spark.functions.geometry import *
from rikai.spark.functions.io import *
from rikai.spark.functions.vision import *

__all__ = ["init"]


def init(spark: SparkSession):
    """Register all rikai UDFs"""
    from rikai.spark.functions import geometry, io, vision

    all_geo_udfs = [
        "area",
        "box2d",
        "box2d_from_center",
        "box2d_from_top_left",
    ]
    all_io_udfs = ["copy"]
    all_vis_udfs = [
        "crop",
        "to_image",
        "image_copy",
        "numpy_to_image",
        "video_to_images",
        "spectrogram_image",
        "video_metadata",
    ]
    for name in all_geo_udfs:
        spark.udf.register(name, getattr(geometry, name))
    for name in all_io_udfs:
        spark.udf.register(name, getattr(io, name))
    for name in all_vis_udfs:
        spark.udf.register(name, getattr(vision, name))
