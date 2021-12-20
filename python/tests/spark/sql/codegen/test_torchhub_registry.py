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

from pyspark.sql import SparkSession

from rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn import (
    OUTPUT_SCHEMA,
)


def test_create_model(rikai_spark: SparkSession):
    rikai_spark.sql(
        f"""
CREATE MODEL resnet50
FLAVOR pytorch
PREPROCESSOR 'rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.pre_processing'
POSTPROCESSOR 'rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.post_processing'
OPTIONS (min_confidence=0.3, device="gpu", batch_size=32)
RETURNS {OUTPUT_SCHEMA}
USING "torchhub:///pytorch/vision:v0.9.1/resnet50";
    """
    )
    rikai_spark.sql("show models").show()
