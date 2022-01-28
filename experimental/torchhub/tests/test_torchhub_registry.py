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

import py4j
import pytest
import torchvision
from pyspark.sql import SparkSession

from rikai.contrib.torch.detections import OUTPUT_SCHEMA

version = f"v{torchvision.__version__.split('+', maxsplit=1)[0]}"


def test_create_model(spark: SparkSession):
    # TODO: run ml_predict on resnet50
    # rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn
    # does not work for torchhub loaded model
    spark.sql(
        f"""
CREATE MODEL create_resnet50
FLAVOR pytorch
PREPROCESSOR 'rikai.contrib.torchhub.pytorch.vision.resnet50.pre_processing'
POSTPROCESSOR 'rikai.contrib.torchhub.pytorch.vision.resnet50.post_processing'
OPTIONS (min_confidence=0.3, device="cpu", batch_size=32)
RETURNS {OUTPUT_SCHEMA}
USING "torchhub:///pytorch/vision:{version}/resnet50";
    """
    )
    assert spark.sql("show models").count() > 0


def test_bad_uri(spark: SparkSession):
    with pytest.raises(py4j.protocol.Py4JJavaError, match=r".*Bad URI.*"):
        spark.sql(
            f"""
CREATE MODEL resnet50_bad_case_1
FLAVOR pytorch
PREPROCESSOR 'rikai.contrib.torchhub.pytorch.vision.resnet50.pre_processing'
POSTPROCESSOR 'rikai.contrib.torchhub.pytorch.vision.resnet50.post_processing'
OPTIONS (min_confidence=0.3, device="cpu", batch_size=32)
RETURNS {OUTPUT_SCHEMA}
USING "torchhub:///pytorch/vision:{version}/resnet50/bad";
        """
        )

    with pytest.raises(py4j.protocol.Py4JJavaError, match=r".*Bad URI.*"):
        spark.sql(
            f"""
CREATE MODEL resnet50_bad_case_2
FLAVOR pytorch
PREPROCESSOR 'rikai.contrib.torchhub.pytorch.vision.resnet50.pre_processing'
POSTPROCESSOR 'rikai.contrib.torchhub.pytorch.vision.resnet50.post_processing'
OPTIONS (min_confidence=0.3, device="cpu", batch_size=32)
RETURNS {OUTPUT_SCHEMA}
USING "torchhub:///pytorch/vision:{version}";
        """
        )

    with pytest.raises(
        py4j.protocol.Py4JJavaError,
        match=r".*URI with 2 forward slashes is not supported.*",
    ):
        spark.sql(
            f"""
CREATE MODEL resnet50_bad_case_3
FLAVOR pytorch
PREPROCESSOR 'rikai.contrib.torchhub.pytorch.vision.resnet50.pre_processing'
POSTPROCESSOR 'rikai.contrib.torchhub.pytorch.vision.resnet50.post_processing'
OPTIONS (min_confidence=0.3, device="cpu", batch_size=32)
RETURNS {OUTPUT_SCHEMA}
USING "torchhub://pytorch/vision:{version}/model_name";
        """
        )
