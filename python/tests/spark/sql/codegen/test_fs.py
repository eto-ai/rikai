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

import secrets
import uuid
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
import torch
import yaml
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import IntegerType, StructField, StructType
from torch.utils.data import DataLoader
from utils import check_ml_predict

from rikai.spark.sql.codegen.fs import FileModelSpec
from rikai.spark.sql.exceptions import SpecError
from rikai.torch.pandas import PandasDataset


def spec_file(content: Dict[str, Any], tmp_path: Path) -> Path:
    filename = f"{secrets.token_urlsafe(6)}.yml"
    spec_filepath = tmp_path / filename
    with spec_filepath.open(mode="w") as fobj:
        yaml.dump(content, fobj)
    return spec_filepath


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
schema: STRUCT<boxes:ARRAY<ARRAY<float>>, scores:ARRAY<float>, labels:ARRAY<int>>
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


@pytest.fixture(scope="module")
def count_objects_spec(tmp_path_factory, resnet_model_uri):
    # Can not use default pytest fixture `tmp_dir` or `tmp_path` because
    # they do not work with module scoped fixture.
    tmp_path = tmp_path_factory.mktemp(str(uuid.uuid4()))
    spec_yaml = """
version: "1.0"
name: resnet
model:
  uri: {}
  flavor: pytorch
schema: int
transforms:
  pre: rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.pre_processing
  post: rikai.testing.predicts.fasterrcnn_resnet_object_counts
    """.format(  # noqa: E501
        resnet_model_uri
    )

    spec_file = tmp_path / "spec.yaml"
    with spec_file.open("w") as fobj:
        fobj.write(spec_yaml)
    yield spec_file


def assert_dataloader_transform(transform):
    data = [1, 1, 1]
    df = pd.DataFrame(data)
    dataset = PandasDataset(df, transform)

    for batch in DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
    ):
        assert batch[0] == torch.tensor([1])


def test_validate_yaml_spec(tmp_path):
    spec = FileModelSpec(
        spec_file(
            {
                "version": "1.2",
                "name": "test_yaml_model",
                "schema": "long",
                "model": {
                    "uri": "s3://bucket/to/model.pt",
                    "unspecified_field": True,
                },
                "options": {"gpu": "true", "batch_size": 123},
            },
            tmp_path,
        )
    )

    assert spec.name == "test_yaml_model"
    assert spec.model_uri == "s3://bucket/to/model.pt"
    assert spec.pre_processing is not None
    assert spec.post_processing is not None
    assert_dataloader_transform(spec.pre_processing)
    assert_dataloader_transform(spec.post_processing)


def test_validate_misformed_spec(tmp_path):
    with pytest.raises(SpecError):
        FileModelSpec(spec_file({}, tmp_path))

    with pytest.raises(SpecError, match=".*version' is a required property.*"):
        FileModelSpec(
            spec_file(
                {
                    "name": "test_yaml_model",
                    "schema": "long",
                    "model": {"uri": "s3://foo/bar"},
                },
                tmp_path,
            )
        )

    with pytest.raises(SpecError, match=".*'model' is a required property.*"):
        FileModelSpec(
            spec_file(
                {
                    "version": "1.0",
                    "name": "test_yaml_model",
                    "schema": "long",
                },
                tmp_path,
            )
        )

    with pytest.raises(SpecError, match=".*'uri' is a required property.*"):
        FileModelSpec(
            spec_file(
                {
                    "version": "1.0",
                    "name": "test_yaml_model",
                    "schema": "long",
                    "model": {},
                },
                tmp_path,
            )
        )


def test_construct_spec_with_options(tmp_path):
    spec = FileModelSpec(
        spec_file(
            {
                "version": "1.0",
                "name": "with_options",
                "schema": "int",
                "model": {
                    "uri": "s3://bucket/to/model.pt",
                    "unspecified_field": True,
                },
            },
            tmp_path,
        ),
        options={"foo": 1, "bar": "2.3"},
    )
    assert {"foo": 1, "bar": "2.3"} == spec.options
    assert "s3://bucket/to/model.pt" == spec.model_uri


@pytest.mark.timeout(60)
def test_yaml_model(spark: SparkSession, resnet_spec: str):
    spark.sql("CREATE MODEL resnet_m USING 'file://{}'".format(resnet_spec))
    check_ml_predict(spark, "resnet_m")


@pytest.mark.timeout(60)
def test_count_objects_model(spark: SparkSession, count_objects_spec: str):
    spark.sql(
        "CREATE MODEL count_objects USING 'file://{}'".format(
            count_objects_spec
        )
    )
    df = spark.createDataFrame(
        [
            # http://cocodataset.org/#explore?id=484912
            Row(
                uri="http://farm2.staticflickr.com/1129/4726871278_4dd241a03a_z.jpg"  # noqa
            ),
            # https://cocodataset.org/#explore?id=433013
            Row(
                uri="http://farm4.staticflickr.com/3726/9457732891_87c6512b62_z.jpg"  # noqa
            ),
        ],
    )
    df.createOrReplaceTempView("df")

    predictions = spark.sql(
        "SELECT ML_PREDICT(count_objects, uri) as objects FROM df"
    )
    assert predictions.schema == StructType(
        [StructField("objects", IntegerType())]
    )
    assert predictions.count() == 2
    assert predictions.where("objects > 0").count() == 2


def test_relative_model_uri(tmp_path):
    spec = FileModelSpec(
        spec_file(
            {
                "version": "1.2",
                "name": "test_yaml_model",
                "schema": "long",
                "model": {"uri": "model.pt"},
            },
            tmp_path,
        )
    )
    assert Path(spec.model_uri) == tmp_path / "model.pt"
