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

import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

try:
    import mlflow
except ImportError:
    raise ImportError(
        "Couldn't import mlflow. Please make sure to "
        "`pip install mlflow` explicitly or install "
        "the correct extras like `pip install rikai[mlflow]`"
    )
from mlflow.tracking import MlflowClient

from rikai.logging import logger
from rikai.spark.sql.codegen.base import ModelSpec, Registry
from rikai.spark.sql.codegen.mlflow_logger import (
    CONF_MLFLOW_LABEL_FUNC,
    CONF_MLFLOW_LABEL_URI,
    CONF_MLFLOW_MODEL_FLAVOR,
    CONF_MLFLOW_MODEL_TYPE,
    CONF_MLFLOW_OUTPUT_SCHEMA,
    CONF_MLFLOW_SPEC_VERSION,
    CONF_MLFLOW_TRACKING_URI,
    MlflowLogger,
)
from rikai.spark.sql.exceptions import SpecError

__all__ = ["MlflowRegistry"]


class MlflowModelSpec(ModelSpec):
    """Model Spec.

    Parameters
    ----------
    model_uri: str
        The uri that Mlflow registry knows how to read
    model_conf: dict
        Configurations required to specify the model
    tracking_uri: str
        The mlflow tracking uri
    options: Dict[str, Any], optional
        Additionally model runtime options
    validate: bool, default True.
        Validate the spec during construction. Default ``True``.
    """

    def __init__(
        self,
        model_uri: str,
        model_conf: dict,
        tracking_uri: str,
        options: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ):
        self.tracking_uri = tracking_uri
        spec_dict = self._load_spec_dict(model_uri, model_conf, options or {})
        super().__init__(spec_dict, validate=validate)
        self._artifact = None

    def load_model(self) -> Any:
        """Load the model artifact specified in this spec"""
        # Currently mlflow model registry load_model is only accessible
        # via fluent API `mlflow.<flavor>.load_model`. So let's be a good
        # samaritan and put the tracking uri back where we found it
        # (just in case)
        old_uri = mlflow.get_tracking_uri()
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            if self.flavor == "tensorflow":
                return mlflow.pyfunc.load_model(
                    self.model_uri
                )._model_impl.model
            else:
                return getattr(mlflow, self.flavor).load_model(self.model_uri)
        finally:
            mlflow.set_tracking_uri(old_uri)

    def _load_spec_dict(self, uri: str, conf: dict, options: dict) -> dict:
        """Convert the Run into a ModelSpec

        Parameters
        ----------
        uri: str
            Rikai model reference
        conf: dict
            Configurations that specifies the model
        options: dict, default None
            Runtime options to be used by the model/transforms

        Returns
        -------
        spec: Dict[str, Any]
        """
        spec = {
            "version": conf.get(
                CONF_MLFLOW_SPEC_VERSION,
                MlflowLogger._CURRENT_MODEL_SPEC_VERSION,
            ),
            "schema": conf.get(CONF_MLFLOW_OUTPUT_SCHEMA, None),
            "model": {
                "flavor": _get_model_prop(conf, CONF_MLFLOW_MODEL_FLAVOR),
                "uri": uri,
                "type": conf.get(CONF_MLFLOW_MODEL_TYPE, None),
            },
            "labels": {
                "func": conf.get(CONF_MLFLOW_LABEL_FUNC, None),
                "uri": conf.get(CONF_MLFLOW_LABEL_URI, None),
            },
        }

        # remove none value
        if not spec["model"]["type"]:
            del spec["model"]["type"]
        if not spec["schema"]:
            del spec["schema"]

        # options
        for key, value in conf.items():
            key = key.lower().strip()
            if key.startswith("rikai.option."):
                sub_len = len("rikai.option.")
                options[key[sub_len:]] = value
        if options:
            options.update(options)
        if len(options) > 0:
            spec["options"] = options

        return spec


def _get_model_prop(
    run_tags: dict,
    conf_name: str,
    extra_options: dict = None,
    option_key: str = None,
    default_value: Any = None,
    raise_if_absent: bool = True,
) -> Any:
    extra_options = extra_options or {}
    option_key = option_key or conf_name
    value = run_tags.get(
        conf_name, extra_options.get(option_key, default_value)
    )
    if not value and raise_if_absent:
        raise ValueError(
            (
                "Please use rikai.mlflow.<flavor>.log_model after "
                "training, or specify {} in CREATE MODEL OPTIONS. "
                "Tags: {}. "
                "Options: {}"
            ).format(conf_name, run_tags, extra_options)
        )
    return value


class MlflowRegistry(Registry):
    """MLFlow-based Model Registry"""

    def __init__(self):
        self._mlflow_client = None

    def __repr__(self):
        return "MlflowRegistry"

    @property
    def tracking_client(self):
        if (
            not self._mlflow_client
            or self._mlflow_client._tracking_client.tracking_uri
            != self.mlflow_tracking_uri
        ):
            self._mlflow_client = MlflowClient(self.mlflow_tracking_uri)
        return self._mlflow_client

    @property
    def mlflow_tracking_uri(self):
        return os.environ.get(CONF_MLFLOW_TRACKING_URI)

    def make_model_spec(self, raw_spec):
        uri = raw_spec["uri"]
        parsed = urlparse(uri)
        if parsed.netloc:
            raise ValueError(
                "URI with 2 forward slashes is not supported, "
                "try URI with 1 slash instead"
            )
        if parsed.scheme != "mlflow":
            raise ValueError("Expect schema: mlflow, but got {parsed.scheme}")
        parts = parsed.path.strip("/").split("/", 1)
        model_uri, run = self.get_model_version(*parts)
        spec = MlflowModelSpec(
            model_uri,
            self.get_model_conf(raw_spec, run),
            self.mlflow_tracking_uri,
            options=self.get_options(raw_spec, run),
        )
        return spec

    def get_model_conf(self, spec, run):
        """
        Get the configurations needed to specify the model
        """
        from_spec = [
            (CONF_MLFLOW_MODEL_FLAVOR, spec["flavor"]),
            (CONF_MLFLOW_OUTPUT_SCHEMA, spec["schema"]),
        ]
        tags = {k: v for k, v in from_spec if v}
        # PEP 448 syntax, right-to-left priority order
        return {**run.data.tags, **tags}

    def get_options(self, spec, run):
        options = run.data.params
        if spec.get("options", {}):
            options.update(spec.get("options"))
        return options

    def get_model_version(
        self, model, stage_or_version=None
    ) -> (str, mlflow.entities.Run):
        """
        Get the model uri that mlflow model registry understands for loading
        a model and the corresponding Run with metadata needed for the spec
        """
        # TODO allow default stage from config
        stage_or_version = stage_or_version or "none"

        if stage_or_version.isdigit():
            # Pegged to version number
            run_id = self.tracking_client.get_model_version(
                model, int(stage_or_version)
            ).run_id
            version = int(stage_or_version)
        else:
            # Latest version in stage
            results = self.tracking_client.get_latest_versions(
                model, stages=[stage_or_version.lower()]
            )
            if not results:
                msg = "No versions found for model {} in stage {}".format(
                    model, stage_or_version
                )
                raise SpecError(msg)
            run_id, version = results[0].run_id, results[0].version

        run = self.tracking_client.get_run(run_id)
        return "models:/{}/{}".format(model, version), run
