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
    def artifact_root_uri(self):
        return self._run.info.artifact_uri

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
        if "rikai.model.artifact.uri" in tags:
            spec["model"]["uri"] = tags["rikai.model.artifact.uri"]
        else:
            # /model/MLModel is currently the MLFlow model schema
            spec["model"]["uri"] = os.path.join(
                self._run.info.artifact_uri, "model/MLModel"
            )
        return spec


def codegen_from_runid(
    spark: SparkSession,
    run_id: str,
    name: Optional[str] = None,
    options: Optional[Dict[str, str]] = None,
) -> str:
    """Generate code from an MLFlow runid

    Parameters
    ----------
    spark : SparkSession
        A live spark session
    run_id : str
        the mlflow runid corresponding to the model spec
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
        import mlflow
    except ImportError:
        raise ImportError(
            "Couldn't import mlflow. Please make sure to "
            "`pip install mlflow` explicitly or install "
            "the correct extras like `pip install rikai[all]`"
        )

    get_run_msg = (
        "Could not get run for {}. Please make sure tracking url "
        "is set properly and the run_id exists."
    ).format(run_id)
    run = _try(mlflow.get_run, get_run_msg, run_id=run_id)

    to_spec_msg = (
        "Could not create well-formed ModelSpec from run {}."
        "Check MLFlowModelSpec._run_to_spec_dict"
    ).format(run_id)
    spec = _try(MlflowModelSpec, to_spec_msg, run, options=options)
    udf = udf_from_spec(spec)
    return register_udf(spark, udf, name)


def _try(func, msg, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        raise SpecError(msg) from e


class MLFlowRegistry(Registry):
    """MLFlow-based Model Registry"""

    def __init__(self, spark: SparkSession):
        self._spark = spark
        self._jvm = spark.sparkContext._jvm

    def __repr__(self):
        return "FileSystemRegistry"

    def resolve(self, uri: str, name: str, options: Dict[str, str]):
        logger.info(f"Resolving model {name} from {uri}")
        # TODO support both reference to a run directly or a
        #  registry model version
        func_name = codegen_from_runid(self._spark, uri, name, options)
        model = self._jvm.ai.eto.rikai.sql.model.mlflow.MLFlowModel(
            name, uri, func_name
        )
        # TODO: set options
        return model
