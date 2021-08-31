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
from pathlib import Path
import rikai
import torch


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
