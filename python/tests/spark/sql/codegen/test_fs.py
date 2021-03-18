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

import tempfile
from pathlib import Path

import pytest
import torch
import torchvision
import numpy as np
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import BinaryType, StructField, StructType

from rikai.spark.sql.codegen.fs import ModelSpec
from rikai.spark.sql.exceptions import SpecError


@pytest.fixture(scope="module")
def yaml_spec():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Prepare model
        resnet = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            progress=False,
        )
        model_uri = tmp_path / "resnet.pth"
        torch.save(resnet, model_uri)

        spec_yaml = """
version: 1.0
name: resnet
model:
    uri: {}
    flavor: pytorch
schema: struct<boxes:array<array<float>>, scores:array<float>, labels:array<int>>
        """.format(  # noqa: E501
            model_uri
        )

        spec_file = tmp_path / "spec.yaml"
        with spec_file.open("w") as fobj:
            fobj.write(spec_yaml)
        yield spec_file


def test_validate_yaml_spec():
    ModelSpec(
        {
            "version": 1.2,
            "name": "test_yaml_model",
            "schema": "long",
            "model": {
                "uri": "s3://bucket/to/model.pt",
                "unspecified_field": True,
            },
            "options": {
                "gpu": "true",
                "batch_size": 123,
            },
        },
    )


def test_validate_misformed_spec():
    with pytest.raises(SpecError):
        ModelSpec({})

    with pytest.raises(SpecError, match=".*version' is a required property.*"):
        ModelSpec(
            {
                "name": "test_yaml_model",
                "schema": "long",
                "model": {"uri": "s3://foo/bar"},
            },
        )

    with pytest.raises(SpecError, match=".*'model' is a required property.*"):
        ModelSpec(
            {
                "version": 1.0,
                "name": "test_yaml_model",
                "schema": "long",
            },
        )

    with pytest.raises(SpecError, match=".*'uri' is a required property.*"):
        ModelSpec(
            {
                "version": 1.0,
                "name": "test_yaml_model",
                "schema": "long",
                "model": {},
            },
        )


def test_yaml_model(spark: SparkSession, yaml_spec):

    spark.sql("CREATE MODEL resnet_m USING 'file://{}'".format(yaml_spec))

    df = spark.createDataFrame(
        [
            Row(
                data=bytearray(
                    np.empty((128, 128, 3), dtype=np.uint8).tobytes()
                )
            )
        ],
        schema=StructType([StructField("data", BinaryType())]),
    )
    df.createOrReplaceTempView("df")

    spark.sql(
        "SELECT ML_PREDICT(resnet_m, data) as predictions FROM df"
    ).show()
