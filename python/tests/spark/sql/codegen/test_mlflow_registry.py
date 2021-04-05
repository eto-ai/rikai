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
import sqlite3
import uuid

import mlflow
import pytest
import torch
from mlflow.tracking import MlflowClient
from pyspark.sql import Row, SparkSession
from utils import check_ml_predict

from rikai.spark.sql.codegen.mlflow_registry import log_model, MlflowModelSpec
from rikai.spark.sql.schema import parse_schema


@pytest.fixture(scope="module")
def mlflow_client(
    tmp_path_factory, resnet_model_uri: str, spark: SparkSession
):
    db_uri = str(tmp_path_factory.mktemp("mlflow") / "mlruns.db")
    tracking_uri = "sqlite:///" + db_uri
    mlflow.set_tracking_uri(tracking_uri)
    # simpliest
    with mlflow.start_run():
        mlflow.log_param("optimizer", "Adam")
        # Fake training loop
        model = torch.load(resnet_model_uri)
        artifact_path = "model"

        schema = (
            "struct<boxes:array<array<float>>,"
            "scores:array<float>, labels:array<int>>"
        )
        pre_processing = (
            "rikai.contrib.torch.transforms."
            "fasterrcnn_resnet50_fpn.pre_processing"
        )
        post_processing = (
            "rikai.contrib.torch.transforms."
            "fasterrcnn_resnet50_fpn.post_processing"
        )
        log_model(
            model,
            artifact_path,
            "pytorch",
            schema,
            pre_processing,
            post_processing,
            registered_model_name="test",
        )
    spark.conf.set("rikai.sql.ml.registry.mlflow.tracking_uri", tracking_uri)
    return mlflow.tracking.MlflowClient(tracking_uri)


def test_modelspec(mlflow_client, resnet_model_uri):
    run_id = mlflow_client.search_model_versions("name='test'")[0].run_id
    run = mlflow_client.get_run(run_id=run_id)
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
    assert spec.uri == "runs://" + run_id + "/model"


@pytest.mark.timeout(60)
def test_mlflow_model_from_runid(
    spark: SparkSession, mlflow_client: MlflowClient
):
    run_id = mlflow_client.search_model_versions("name='test'")[0].run_id
    spark.sql("CREATE MODEL resnet_m_foo USING 'mlflow://{}'".format(run_id))
    check_ml_predict(spark, "resnet_m_foo")


@pytest.mark.timeout(60)
def test_mlflow_model_from_model_version(spark: SparkSession, mlflow_client):
    spark.sql("CREATE MODEL resnet_m_bar USING 'mlflow://test/1'")
    check_ml_predict(spark, "resnet_m_bar")
