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

import uuid

import pytest
from pyspark.sql import Row, SparkSession
from utils import check_ml_predict

from rikai.spark.sql.codegen.fs import FileModelSpec
from rikai.spark.sql.exceptions import SpecError


@pytest.fixture(scope="module")
def resnet_spec(tmp_path_factory, resnet_model_uri):
    # Can not use default pytest fixture `tmp_dir` or `tmp_path` because
    # they do not work with module scoped fixture.
    tmp_path = tmp_path_factory.mktemp(str(uuid.uuid4()))
    spec_yaml = """
version: "1.0"
name: resnet
model:
  uri: {}
  flavor: pytorch
schema: struct<boxes:array<array<float>>, scores:array<float>, labels:array<int>>
transforms:
  pre: rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.pre_processing
  post: rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.post_processing
    """.format(  # noqa: E501
        resnet_model_uri
    )

    spec_file = tmp_path / "spec.yaml"
    with spec_file.open("w") as fobj:
        fobj.write(spec_yaml)
    yield spec_file


def test_validate_yaml_spec():
    spec = FileModelSpec(
        {
            "version": "1.2",
            "name": "test_yaml_model",
            "schema": "long",
            "model": {
                "uri": "s3://bucket/to/model.pt",
                "unspecified_field": True,
            },
            "options": {"gpu": "true", "batch_size": 123},
        }
    )

    assert spec.name == "test_yaml_model"
    assert spec.pre_processing is not None
    assert spec.post_processing is not None


def test_validate_misformed_spec():
    with pytest.raises(SpecError):
        FileModelSpec({})

    with pytest.raises(SpecError, match=".*version' is a required property.*"):
        FileModelSpec(
            {
                "name": "test_yaml_model",
                "schema": "long",
                "model": {"uri": "s3://foo/bar"},
            }
        )

    with pytest.raises(SpecError, match=".*'model' is a required property.*"):
        FileModelSpec(
            {"version": "1.0", "name": "test_yaml_model", "schema": "long"}
        )

    with pytest.raises(SpecError, match=".*'uri' is a required property.*"):
        FileModelSpec(
            {
                "version": "1.0",
                "name": "test_yaml_model",
                "schema": "long",
                "model": {},
            }
        )


def test_construct_spec_with_options():
    spec = FileModelSpec(
        {
            "version": "1.0",
            "name": "with_options",
            "schema": "int",
            "model": {
                "uri": "s3://bucket/to/model.pt",
                "unspecified_field": True,
            },
        },
        options={"foo": 1, "bar": "2.3"},
    )
    assert {"foo": 1, "bar": "2.3"} == spec.options


@pytest.mark.timeout(60)
def test_yaml_model(spark: SparkSession, resnet_spec: str):
    spark.sql("CREATE MODEL resnet_m USING 'file://{}'".format(resnet_spec))
    check_ml_predict(spark, "resnet_m")
