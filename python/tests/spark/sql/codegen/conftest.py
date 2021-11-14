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

from __future__ import annotations

import mlflow
import pytest
import torch
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

import rikai
from rikai.spark.utils import get_default_jar_version, init_spark_session


@pytest.fixture(scope="module")
def mlflow_client(
    tmp_path_factory, resnet_model_uri: str, spark: SparkSession
) -> MlflowClient:
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
            "STRUCT<boxes:ARRAY<ARRAY<float>>,"
            "scores:ARRAY<float>,label_ids:ARRAY<int>>"
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
        mlflow.pytorch.log_model(
            model, artifact_path, registered_model_name="vanilla-mlflow"
        )
        mlflow.set_tags(
            {
                "rikai.model.flavor": "pytorch",
                "rikai.output.schema": schema,
                "rikai.transforms.pre": pre_processing,
                "rikai.transforms.post": post_processing,
            }
        )

    # vanilla mlflow no tags
    with mlflow.start_run():
        mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name="vanilla-mlflow-no-tags",
        )

    # vanilla mlflow wrong tags
    with mlflow.start_run():
        mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name="vanilla-mlflow-wrong-tags",
        )
        mlflow.set_tags(
            {
                "rikai.model.flavor": "pytorch",
                "rikai.output.schema": schema,
                "rikai.transforms.pre": "wrong_pre",
                "rikai.transforms.post": "wrong_post",
            }
        )

    spark.conf.set("rikai.sql.ml.registry.mlflow.tracking_uri", tracking_uri)
    return mlflow.tracking.MlflowClient(tracking_uri)


@pytest.fixture()
def spark_with_mlflow(mlflow_client) -> SparkSession:
    mlflow_tracking_uri = mlflow.get_tracking_uri()
    active_session = SparkSession.getActiveSession()
    if (
        active_session
        and active_session.conf.get("rikai.sql.ml.catalog.impl", None)
        != "ai.eto.rikai.sql.model.mlflow.MlflowCatalog"
    ):
        active_session.stop()

    rikai_version = get_default_jar_version(use_snapshot=True)

    spark = init_spark_session(
        conf=dict(
            [
                (
                    "spark.jars.packages",
                    ",".join(
                        [
                            "ai.eto:rikai_2.12:{}".format(rikai_version),
                        ]
                    ),
                ),
                (
                    "rikai.sql.ml.catalog.impl",
                    "ai.eto.rikai.sql.model.mlflow.MlflowCatalog",
                ),
                (
                    "rikai.sql.ml.registry.mlflow.tracking_uri",
                    mlflow_tracking_uri,
                ),
            ]
        )
    )
    yield spark

    try:
        for model in mlflow_client.list_registered_models():
            print(f"Cleanup {model.name}")
            mlflow_client.delete_registered_model(model.name)
        for run in mlflow_client.list_run_infos():
            print(f"Clean run: {run}")
            mlflow_client.delete_run(run.run_id)
    except:
        pass
