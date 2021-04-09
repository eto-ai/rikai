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
"""Custom Mlflow model logger to make sure models have the right logging for
Rikai SQL ML
"""
from typing import Any, Optional

CONF_MLFLOW_TRACKING_URI = "rikai.sql.ml.registry.mlflow.tracking_uri"
CONF_MLFLOW_OUTPUT_SCHEMA = "rikai.output.schema"
CONF_MLFLOW_SPEC_VERSION = "rikai.spec.version"
CONF_MLFLOW_PRE_PROCESSING = "rikai.transforms.pre"
CONF_MLFLOW_POST_PROCESSING = "rikai.transforms.post"
CONF_MLFLOW_MODEL_FLAVOR = "rikai.model.flavor"
CONF_MLFLOW_ARTIFACT_PATH = "rikai.model.artifact_path"


class MlflowLogger:
    """
    An alternative model logger for use during training instead of the vanilla

    """

    _CURRENT_MODEL_SPEC_VERSION = "1.0"

    def __init__(self, flavor: str):
        """
        Parameters
        ----------
        flavor: str
            The model flavor
        """
        self.spec_version = MlflowLogger._CURRENT_MODEL_SPEC_VERSION
        self.flavor = flavor

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        schema: str,
        pre_processing: Optional[str] = None,
        post_processing: Optional[str] = None,
        registered_model_name: Optional[str] = None,
        **kwargs,
    ):
        """Convenience function to log the model with tags needed by rikai.
        This should be called during training when the model is produced.

        Parameters
        ----------
        model: Any
            The model artifact object
        artifact_path: str
            The relative (to the run) artifact path
        schema: str
            Output schema (pyspark DataType)
        pre_processing: str, default None
            Full python module path of the pre-processing transforms
        post_processing: str, default None
            Full python module path of the post-processing transforms
        registered_model_name: str, default None
            Model name in the mlflow model registry
        kwargs: dict
            Passed to `mlflow.<flavor>.log_model`

        Examples
        --------
        # import mlflow.pytorch  <= vanilla mlflow usage
        import rikai.mlflow

        # Log PyTorch model
        with mlflow.start_run() as run:

            # Training loop
            # ...

            # Assume `model` is the trained model from the training loop
            rikai.mlflow.pytorch.log_model(model, "model",
                      schema=rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.OUTPUT_SCHEMA,
                      pre_processing="rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.pre_processing",
                      post_processing="rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.post_processing",
                      registered_model_name="MyPytorchModel")

        For more details see `mlflow docs <https://www.mlflow.org/docs/latest/python_api/mlflow.pytorch.html#mlflow.pytorch.log_model>`_.
        """  # noqa E501

        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "Couldn't import mlflow. Please make sure to "
                "`pip install mlflow` explicitly or install "
                "the correct extras like `pip install rikai[mlflow]`"
            ) from e

        # no need to set the tracking uri here since this is intended to be
        # called inside the training loop within mlflow.start_run
        getattr(mlflow, flavor).log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name,
            **kwargs,
        )
        tags = {
            CONF_MLFLOW_SPEC_VERSION: MlflowLogger._CURRENT_MODEL_SPEC_VERSION,
            CONF_MLFLOW_MODEL_FLAVOR: self.flavor,
            CONF_MLFLOW_OUTPUT_SCHEMA: schema,
            CONF_MLFLOW_PRE_PROCESSING: pre_processing,
            CONF_MLFLOW_POST_PROCESSING: post_processing,
            CONF_MLFLOW_ARTIFACT_PATH: artifact_path,
        }
        mlflow.set_tags(tags)


KNOWN_FLAVORS = ["pytorch"]
for flavor in KNOWN_FLAVORS:
    globals()[flavor] = MlflowLogger(flavor)
