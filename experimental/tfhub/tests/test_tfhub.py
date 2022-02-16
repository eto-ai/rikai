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

from pathlib import Path

from pyspark.sql import SparkSession

from rikai.spark.functions import to_image
from rikai.contrib.tensorflow.models.ssd import OUTPUT_SCHEMA


def test_ssd(spark: SparkSession):
    spark.udf.register("to_image", to_image)
    work_dir = Path().absolute().parent.parent
    image_path = f"{work_dir}/python/tests/assets/test_image.jpg"
    spark.sql(
        f"""
        CREATE MODEL tfssd
        PREPROCESSOR 'rikai.contrib.tensorflow.models.ssd.pre_processing'
        POSTPROCESSOR 'rikai.contrib.tensorflow.models.ssd.post_processing'
        OPTIONS (device="cpu", batch_size=32)
        RETURNS {OUTPUT_SCHEMA}
        USING "tfhub:///tensorflow/ssd_mobilenet_v2/2";
        """
    )
    result = spark.sql(
        f"""
    select ML_PREDICT(tfssd, to_image('{image_path}')) as preds
    """
    )
    assert result.count() == 1
