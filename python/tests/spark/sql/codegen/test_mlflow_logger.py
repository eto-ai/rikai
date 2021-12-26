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

from pathlib import Path

import mlflow
import torch
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

import rikai


def test_none_value_tags(tmp_path: Path, resnet_model_uri):
    tmp_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = "sqlite:///" + str(tmp_path / "tracking.db")
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run():
        model = torch.load(resnet_model_uri)
        rikai.mlflow.pytorch.log_model(
            model,
            "artifact_path",
            "output_schema",
            pre_processing=None,
            post_processing=None,
        )
        run_id = mlflow.active_run().info.run_id
        run = mlflow.get_run(run_id)
        tags = run.data.tags
        assert tags[rikai.mlflow.CONF_MLFLOW_ARTIFACT_PATH] == "artifact_path"
        assert tags[rikai.mlflow.CONF_MLFLOW_OUTPUT_SCHEMA] == "output_schema"
        assert rikai.mlflow.CONF_MLFLOW_PRE_PROCESSING not in tags
        assert rikai.mlflow.CONF_MLFLOW_POST_PROCESSING not in tags


def test_model_and_version_tags(tmp_path: Path, resnet_model_uri):
    tmp_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = "sqlite:///" + str(tmp_path / "tracking.db")
    mlflow.set_tracking_uri(tracking_uri)
    model_name = "model_for_tags"
    c = MlflowClient()
    c.create_registered_model(model_name)
    with mlflow.start_run():
        model = torch.load(resnet_model_uri)
        rikai.mlflow.pytorch.log_model(
            model,
            "artifact_path",
            "output_schema",
            registered_model_name=model_name,
            pre_processing=None,
            post_processing=None,
        )
        run_id = mlflow.active_run().info.run_id
        run = mlflow.get_run(run_id)
        tags = run.data.tags

    logged_model = c.get_registered_model(model_name)
    print(f"current logged model {logged_model}")
    assert (
        logged_model.tags[rikai.mlflow.CONF_MLFLOW_ARTIFACT_PATH]
        == tags[rikai.mlflow.CONF_MLFLOW_ARTIFACT_PATH]
    )
    assert (
        logged_model.tags[rikai.mlflow.CONF_MLFLOW_OUTPUT_SCHEMA]
        == tags[rikai.mlflow.CONF_MLFLOW_OUTPUT_SCHEMA]
    )
    assert (
        logged_model.tags[rikai.mlflow.CONF_MLFLOW_MODEL_FLAVOR] == "pytorch"
    )

    newer_all_versions = c.get_latest_versions(
        model_name, stages=["production", "staging", "None"]
    )
    current_version: ModelVersion = None
    for v in newer_all_versions:
        if v.run_id == run_id:
            current_version = v
            break
    if current_version is None:
        raise ValueError(
            "No model version found matching runid: {}".format(run_id)
        )
    print(f"current version {current_version}")
    assert (
        current_version.tags[rikai.mlflow.CONF_MLFLOW_ARTIFACT_PATH]
        == tags[rikai.mlflow.CONF_MLFLOW_ARTIFACT_PATH]
    )
    assert (
        current_version.tags[rikai.mlflow.CONF_MLFLOW_OUTPUT_SCHEMA]
        == tags[rikai.mlflow.CONF_MLFLOW_OUTPUT_SCHEMA]
    )
    assert (
        current_version.tags[rikai.mlflow.CONF_MLFLOW_MODEL_FLAVOR]
        == "pytorch"
    )

    # verify the tags would be overwritten
    with mlflow.start_run():
        model = torch.load(resnet_model_uri)
        rikai.mlflow.pytorch.log_model(
            model,
            "artifact_path2",
            "output_schema2",
            registered_model_name=model_name,
            pre_processing=None,
            post_processing=None,
        )
        second_run_id = mlflow.active_run().info.run_id
        run = mlflow.get_run(second_run_id)
        second_run_tags = run.data.tags
    assert run_id != second_run_id
    logged_model = c.get_registered_model(model_name)
    print(f"current logged model2 {logged_model}")

    assert (
        logged_model.tags[rikai.mlflow.CONF_MLFLOW_ARTIFACT_PATH]
        == second_run_tags[rikai.mlflow.CONF_MLFLOW_ARTIFACT_PATH]
    )
    assert (
        logged_model.tags[rikai.mlflow.CONF_MLFLOW_OUTPUT_SCHEMA]
        == second_run_tags[rikai.mlflow.CONF_MLFLOW_OUTPUT_SCHEMA]
    )
    assert (
        logged_model.tags[rikai.mlflow.CONF_MLFLOW_MODEL_FLAVOR] == "pytorch"
    )

    newer_all_versions = c.get_latest_versions(
        model_name, stages=["production", "staging", "None"]
    )
    for v in newer_all_versions:
        if v.run_id == second_run_id:
            current_version = v
            break
    if current_version is None:
        raise ValueError(
            "No model version found matching runid: {}".format(second_run_id)
        )
    print(f"current version2 {current_version}")
    assert (
        current_version.tags[rikai.mlflow.CONF_MLFLOW_ARTIFACT_PATH]
        == second_run_tags[rikai.mlflow.CONF_MLFLOW_ARTIFACT_PATH]
    )
    assert (
        current_version.tags[rikai.mlflow.CONF_MLFLOW_OUTPUT_SCHEMA]
        == second_run_tags[rikai.mlflow.CONF_MLFLOW_OUTPUT_SCHEMA]
    )
    assert (
        current_version.tags[rikai.mlflow.CONF_MLFLOW_MODEL_FLAVOR]
        == "pytorch"
    )
