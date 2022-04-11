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

import datetime
import os
import uuid

import mlflow
import pytest
import tensorflow_hub as hub
import tensorflow as tf
import torch
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

import rikai
from rikai.contrib.tfhub.tensorflow.ssd import HUB_URL as SSD_HUB_URL
from rikai.spark.utils import get_default_jar_version, init_spark_session
from rikai.spark.sql.codegen.mlflow_registry import CONF_MLFLOW_TRACKING_URI


@pytest.fixture(scope="session")
def tfhub_ssd(tmp_path_factory):
    m = hub.load(SSD_HUB_URL)
    tmp_path = tmp_path_factory.mktemp(str(uuid.uuid4()))
    model_path = str(tmp_path / "model")
    tf.saved_model.save(m, model_path)
    return (m, model_path)


@pytest.fixture(scope="session")
def mlflow_client_http(
    tmp_path_factory, resnet_model_uri: str
) -> MlflowClient:
    tracking_uri = os.getenv(
        "TEST_MLFLOW_TRACKING_URI", "http://localhost:5000"
    )
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment(
        "rikai-test" + str(datetime.datetime.now()), "test-artifact"
    )
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
        rikai.mlflow.pytorch.log_model(
            model,  # same as vanilla mlflow
            artifact_path,  # same as vanilla mlflow
            schema,
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
    return mlflow.tracking.MlflowClient(tracking_uri)


@pytest.fixture(scope="class")
def spark_with_mlflow(mlflow_client_http) -> SparkSession:
    mlflow_tracking_uri = mlflow.get_tracking_uri()
    print(f"Spark with mlflow tracking uri: ${mlflow_tracking_uri}")
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
                    "spark.rikai.sql.ml.catalog.impl",
                    "ai.eto.rikai.sql.model.mlflow.MlflowCatalog",
                ),
                (
                    CONF_MLFLOW_TRACKING_URI,
                    mlflow_tracking_uri,
                ),
            ]
        ),
        app_name="rikai_with_mlflow",
    )
    yield spark

    try:
        for model in mlflow_client_http.list_registered_models():
            print(f"Cleanup {model.name}")
            mlflow_client_http.delete_registered_model(model.name)
        for run in mlflow_client_http.list_run_infos():
            print(f"Clean run: {run}")
            mlflow_client_http.delete_run(run.run_id)
    except Exception:
        pass
