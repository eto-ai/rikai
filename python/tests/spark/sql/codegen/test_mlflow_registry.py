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

import mlflow
import pytest
from pyspark.sql import Row, SparkSession
from utils import check_ml_predict

from rikai.spark.sql.codegen.mlflow_registry import (
    codegen_from_runid,
    MlflowModelSpec,
)
from rikai.spark.sql.schema import parse_schema


@pytest.fixture(scope="module")
def mlflow_client(tmp_path_factory, resnet_model_uri):
    tracking_uri = (
        "file://" + str(tmp_path_factory.mktemp("mlflow_tracking")) + "/mlruns"
    )
    mlflow.set_tracking_uri(tracking_uri)
    # simpliest
    with mlflow.start_run():
        mlflow.log_param("optimizer", "Adam")
        mlflow.set_tags(
            {
                "rikai.pytest.testcase": "simple",
                "rikai.spec.version": "1.0",
                "rikai.model.flavor": "pytorch",
                "rikai.output.schema": (
                    "struct<boxes:array<array<float>>, "
                    "scores:array<float>, labels:array<int>>"
                ),
                "rikai.transforms.pre": (
                    "rikai.contrib.torch.transforms."
                    "fasterrcnn_resnet50_fpn.pre_processing"
                ),
                "rikai.transforms.post": (
                    "rikai.contrib.torch.transforms."
                    "fasterrcnn_resnet50_fpn."
                    "post_processing"
                ),
                "rikai.model.artifact.uri": resnet_model_uri,
            }
        )
    # TODO use log_model
    # TODO create model version
    return mlflow.tracking.MlflowClient(tracking_uri)


def test_modelspec(mlflow_client, resnet_model_uri):
    run = mlflow_client.search_runs(
        "0", 'tags.rikai.pytest.testcase = "simple"'
    )[0]
    spec = MlflowModelSpec(run)
    assert spec.flavor == "pytorch"
    assert spec.schema == parse_schema(
        (
            "struct<boxes:array<array<float>>, "
            "scores:array<float>, labels:array<int>>"
        )
    )
    assert spec._spec["transforms"]["pre"] == (
        "rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn"
        ".pre_processing"
    )
    assert spec._spec["transforms"]["post"] == (
        "rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn."
        "post_processing"
    )
    assert spec.uri == str(resnet_model_uri)
