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

from rikai.spark.sql.callback_service import init_cb_service

__all__ = ["init"]


def init(spark: SparkSession):
    """Initialize SQL-ML services.

    .. code-block:: python

        # To start use Rikai SQL-ML in Spark
        from rikai.spark.sql import init

        spark = (
            SparkSession
            ...
            .getOrCreate)
        init(spark)

    """
    init_cb_service(spark)

    from rikai.spark.functions.vision import (
        crop,
        image_copy,
        numpy_to_image,
        spectrogram_image,
        to_image,
        video_metadata,
        video_to_images,
    )

    spark.udf.register("crop", crop)
    spark.udf.register("image_copy", image_copy)
    spark.udf.register("numpy_to_image", numpy_to_image)
    spark.udf.register("spectrogram_image", spectrogram_image)
    spark.udf.register("to_image", to_image)
    spark.udf.register("video_metadata", video_metadata)
    spark.udf.register("video_to_images", video_to_images)
