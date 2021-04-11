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
import torch
from mlflow.tracking import MlflowClient
from pyspark.sql import Row, SparkSession
from utils import check_ml_predict

import rikai
from rikai.spark.sql.codegen.mlflow_registry import MlflowModelSpec
from rikai.spark.sql.schema import parse_schema


@pytest.fixture(scope="module")
def mlflow_client(
    tmp_path_factory, resnet_model_uri: str, spark: SparkSession
):
    tmp_path = tmp_path_factory.mktemp("mlflow")
    tmp_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = "sqlite:///" + str(tmp_path / "tracking.db")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment("rikai-test", str(tmp_path))
    # simpliest
    with mlflow.start_run(experiment_id=experiment_id):
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
        rikai.mlflow.pytorch.log_model(
            model,  # same as vanilla mlflow
            artifact_path,  # same as vanilla mlflow
            schema,
            pre_processing,
            post_processing,
            registered_model_name="rikai-test",  # same as vanilla mlflow
        )

    # vanilla mlflow
    with mlflow.start_run():
        mlflow.pytorch.log_model(model, artifact_path,
                                 registered_model_name="vanilla-mlflow")
    spark.conf.set("rikai.sql.ml.registry.mlflow.tracking_uri", tracking_uri)
    return mlflow.tracking.MlflowClient(tracking_uri)


def test_modelspec(mlflow_client, resnet_model_uri):
    run_id = mlflow_client.search_model_versions("name='rikai-test'")[0].run_id
    run = mlflow_client.get_run(run_id=run_id)
    spec = MlflowModelSpec(
        "runs:/{}/model".format(run_id),
        run.data.tags,
        run.data.params,
        tracking_uri="fake",
    )
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
    assert spec.uri == "runs:/" + run_id + "/model"


@pytest.mark.timeout(60)
def test_mlflow_model_from_runid(
    spark: SparkSession, mlflow_client: MlflowClient
):
    run_id = mlflow_client.search_model_versions("name='rikai-test'")[0].run_id

    spark.sql(
        "CREATE MODEL resnet_m_foo USING 'mlflow://{}/model'".format(run_id)
    )
    check_ml_predict(spark, "resnet_m_foo")

    # if no path is given but only one artifact exists then use it by default
    spark.sql("CREATE MODEL resnet_m_bar USING 'mlflow://{}'".format(run_id))
    check_ml_predict(spark, "resnet_m_bar")


@pytest.mark.timeout(60)
def test_mlflow_model_from_model_version(spark: SparkSession, mlflow_client):
    # peg to a particular version of a model
    spark.sql("CREATE MODEL resnet_m_fizz USING 'mlflow://rikai-test/1'")
    check_ml_predict(spark, "resnet_m_fizz")

    # use the latest version in a given stage (omitted means none)
    spark.sql("CREATE MODEL resnet_m_buzz USING 'mlflow://rikai-test/'")
    check_ml_predict(spark, "resnet_m_buzz")


@pytest.mark.timeout(60)
def test_mlflow_model_without_custom_logger(spark: SparkSession, mlflow_client):
    # peg to a particular version of a model
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

    sql = ('CREATE MODEL vanilla_ice ' 
           'OPTIONS ('
           '"rikai.model.flavor"="pytorch",'
           '"rikai.output.schema"="{}",'
           '"rikai.transforms.pre"="{}",'
           '"rikai.transforms.post"="{}") '
           'USING "mlflow://vanilla-mlflow/1"').format(schema, pre_processing,
                                                       post_processing)
    spark.sql(sql)
    check_ml_predict(spark, "vanilla_ice")

