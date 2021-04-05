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
from typing import Any, Callable, Dict, IO, Mapping, Optional, Union
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
from mlflow.exceptions import MlflowException
from pyspark.sql import SparkSession

from rikai.logging import logger
from rikai.spark.sql.codegen.base import (
    ModelSpec,
    register_udf,
    Registry,
    udf_from_spec,
)
from rikai.spark.sql.exceptions import SpecError

__all__ = ["MLFlowRegistry"]


class MlflowModelSpec(ModelSpec):
    """Model Spec.

    Parameters
    ----------
    run : mlflow.entities.Run
        the Run object containing the spec info and ref to model artifact
    options : Dict[str, Any], optional
        Additionally options. If the same option exists in spec already,
        it will be overridden.
    validate : bool, default True.
        Validate the spec during construction. Default ``True``.
    """

    def __init__(
        self,
        run: "mlflow.entities.Run",
        options: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ):
        self._run = run
        spec = self._run_to_spec_dict()
        spec.setdefault("options", {})
        if options:
            spec["options"].update(options)
        super().__init__(spec, validate=validate)
        self._artifact = None

    @property
    def run_id(self):
        return self._run.info.run_id

    @property
    def start_time(self):
        return self._run.info.start_time

    @property
    def end_time(self):
        return self._run.info.end_time

    @property
    def artifact(self) -> Any:
        """Return Model artifact"""
        if self._artifact is None:
            self._artifact = getattr(mlflow, self.flavor).load_model(
                'runs://' + self.run_id + '/' + self.uri)
        return self._artifact

    def _run_to_spec_dict(self):
        """Convert the Run into a ModelSpec

        Returns
        -------
        spec: Dict[str, Any]
        """
        tags = self._run.data.tags
        spec = {
            "version": tags["rikai.spec.version"],
            "schema": tags["rikai.output.schema"],
            "model": {"flavor": tags["rikai.model.flavor"]},
        }

        # transforms
        transforms = {}
        if "rikai.transforms.pre" in tags:
            transforms["pre"] = tags["rikai.transforms.pre"]
        if "rikai.transforms.post" in tags:
            transforms["post"] = tags["rikai.transforms.post"]
        if len(transforms) > 0:
            spec["transforms"] = transforms

        # options
        options = dict(self._run.data.params or {})  # for training
        for key, value in tags.items():
            key = key.lower().strip()
            if key.startswith("rikai.option."):
                options[key[len("rikai.option.") :]] = value
        if len(options) > 0:
            spec["options"] = options

        # uri (to the model artifact)
        if "rikai.model.artifact_path" in tags:
            spec["model"]["uri"] = os.path.join(
                'runs://', self._run.info.run_id,
                tags["rikai.model.artifact_path"])
        return spec


def codegen_from_run(
    spark: SparkSession,
    run: mlflow.entities.Run,
    name: Optional[str] = None,
    options: Optional[Dict[str, str]] = None,
) -> str:
    """Generate code from an MLFlow runid

    Parameters
    ----------
    spark : SparkSession
        A live spark session
    run : mlflow.entities.Run
        the mlflow run that produced the model we want to register
    name : str
        The name of the model in the catalog
    options : dict
        Optional parameters passed to the model.

    Returns
    -------
    str
        Spark UDF function name for the generated data.
    """
    try:
        spec = MlflowModelSpec(run, options=options)
    except Exception:
        to_spec_msg = (
            "Could not create well-formed ModelSpec from run {}."
            "Check MLFlowModelSpec._run_to_spec_dict"
        ).format(run_id)
        raise SpecError(to_spec_msg) from e
    udf = udf_from_spec(spec)
    return register_udf(spark, udf, name)


def log_model(model: Any, artifact_path: str, flavor: str, schema: str,
              pre_processing: Optional[str] = None,
              post_processing: Optional[str] = None,
              registered_model_name: Optional[str] = None,
              **kwargs):
    """Convenience function to log the model with information needed by rikai

    Parameters
    ----------
    model: Any
        The model artifact object
    artifact_path: str
        The relative (to the run) artifact path
    flavor: str
        A valid Mlflow flavor (e.g., "pytorch")
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
    """
    # no need to set the tracking uri here since this is intended to be called
    # inside the training loop within mlflow.start_run
    getattr(mlflow, flavor).log_model(
        model, artifact_path, registered_model_name=registered_model_name,
        **kwargs)
    tags = {
        "rikai.spec.version": "1.0",
        "rikai.model.flavor": flavor,
        "rikai.output.schema": schema,
        "rikai.transforms.pre": pre_processing,
        "rikai.transforms.post": post_processing,
        "rikai.model.artifact_path": artifact_path
    }
    mlflow.set_tags(tags)


class MlflowRegistry(Registry):
    """MLFlow-based Model Registry"""

    def __init__(self, spark: SparkSession):
        self._spark = spark
        self._jvm = spark.sparkContext._jvm

    def __repr__(self):
        return "MlflowRegistry"

    def resolve(self, uri: str, name: str, options: Dict[str, str]):
        logger.info(f"Resolving model {name} from {uri}")
        tracking_uri = spark.conf.get(
            'rikai.sql.ml.registry.mlflow.tracking_uri')
        client = MlflowClient(tracking_uri)
        run = get_run(client, uri)
        func_name = codegen_from_run(self._spark, run, name, options)
        model = self._jvm.ai.eto.rikai.sql.model.mlflow.MlflowModel(
            name, run.run_id, func_name
        )
        return model


def get_run(client: MlflowClient, uri: str) -> mlflow.entities.Run:
    parsed = urlparse(uri)
    runid_or_model = parsed.hostname
    try:
        # we can support a direct reference to runid
        # 'mlflow://<run_id>'
        return client.get_run(runid_or_model)
    except MlflowException:
        if parsed.path and parsed.path[1:].isdigit():
            # we can support a reference to a registered model and version
            # 'mlflow://<model_name>/<version>'
            return client.get_run(client.get_model_version(
                runid_or_model, int(parsed.path[1:])).run_id)

        # Or just use the latest version (by stage)
        if not parsed.path:
            # 'mlflow://<model_name>'
            stage = 'none'
        else:
            # 'mlflow://<model_name>/<stage>'
            stage = parsed.path[1:].lower()
        for x in client.get_registered_model(runid_or_model).latest_versions:
            if x.current_stage.lower() == stage:
                return client.get_run(x.run_id)