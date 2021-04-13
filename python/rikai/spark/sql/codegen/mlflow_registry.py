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

from typing import Any, Callable, Dict, IO, Mapping, Optional, Union
from urllib.parse import ParseResult, urlparse

try:
    import mlflow
except ImportError:
    raise ImportError(
        "Couldn't import mlflow. Please make sure to "
        "`pip install mlflow` explicitly or install "
        "the correct extras like `pip install rikai[mlflow]`"
    )
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

from rikai.logging import logger
from rikai.spark.sql.codegen.base import (
    ModelSpec,
    register_udf,
    Registry,
    udf_from_spec,
)
from rikai.spark.sql.codegen.mlflow_logger import (
    CONF_MLFLOW_ARTIFACT_PATH,
    CONF_MLFLOW_MODEL_FLAVOR,
    CONF_MLFLOW_OUTPUT_SCHEMA,
    CONF_MLFLOW_POST_PROCESSING,
    CONF_MLFLOW_PRE_PROCESSING,
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
    tags: dict
        Tags from the Mlflow run
    params: dict
        Params from the Mlflow run
    tracking_uri: str
        The mlflow tracking uri
    options: Dict[str, Any], optional
        Additionally options. If the same option exists in spec already,
        it will be overridden.
    validate: bool, default True.
        Validate the spec during construction. Default ``True``.
    """

    def __init__(
        self,
        model_uri: str,
        tags: dict,
        params: dict,
        tracking_uri: str,
        options: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ):
        self.tracking_uri = tracking_uri
        spec = self._load_spec_dict(model_uri, tags, params, options)
        super().__init__(spec, validate=validate)
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
            return getattr(mlflow, self.flavor).load_model(self.model_uri)
        finally:
            mlflow.set_tracking_uri(old_uri)

    def _load_spec_dict(
        self, uri: str, tags: dict, params: dict, extras=None
    ) -> dict:
        """Convert the Run into a ModelSpec

        Parameters
        ----------
        uri: str
            Rikai model reference
        tags: dict
            Tags from the Mlflow Run
        params: dict
            Params from the Mlflow Run
        extras: dict, default None
            Extra options passed in at model registration time

        Returns
        -------
        spec: Dict[str, Any]
        """
        extras = extras or {}
        spec = {
            "version": tags.get(
                CONF_MLFLOW_SPEC_VERSION,
                MlflowLogger._CURRENT_MODEL_SPEC_VERSION,
            ),
            "schema": _get_model_prop(tags, CONF_MLFLOW_OUTPUT_SCHEMA),
            "model": {
                "flavor": _get_model_prop(tags, CONF_MLFLOW_MODEL_FLAVOR),
                "uri": uri,
            },
            "transforms": {
                "pre": tags.get(CONF_MLFLOW_PRE_PROCESSING, None),
                "post": tags.get(CONF_MLFLOW_POST_PROCESSING, None),
            },
        }

        # options
        options = dict(params or {})  # for training
        for key, value in tags.items():
            key = key.lower().strip()
            if key.startswith("rikai.option."):
                sub_len = len("rikai.option.")
                options[key[sub_len:]] = value
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


def codegen_from_spec(
    spark: SparkSession, spec: dict, name: Optional[str] = None
) -> str:
    """Generate code from an MLFlow runid

    Parameters
    ----------
    spark : SparkSession
        A live spark session
    spec : dict
        the model spec info dict
    name : str
        The name of the model in the catalog

    Returns
    -------
    str
        Spark UDF function name for the generated data.
    """
    udf = udf_from_spec(spec)
    return register_udf(spark, udf, name)


class MlflowRegistry(Registry):
    """MLFlow-based Model Registry"""

    def __init__(self, spark: SparkSession):
        self._spark = spark
        self._jvm = spark.sparkContext._jvm
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
        return self._spark.conf.get(CONF_MLFLOW_TRACKING_URI)

    def resolve(self, spec):
        name = spec.getName()
        uri = spec.getUri()
        options = spec.getOptions()
        logger.info(f"Resolving model {name} from {uri}")
        model_uri, tags, params = _get_model_info(uri, self.tracking_client)
        spec = MlflowModelSpec(
            model_uri, tags, params, self.mlflow_tracking_uri, options=options
        )
        func_name = codegen_from_spec(self._spark, spec, name)
        model = self._jvm.ai.eto.rikai.sql.model.SparkUDFModel(
            name, uri, func_name
        )
        return model


def _get_model_info(uri: str, client: MlflowClient) -> str:
    """Transform the rikai model uri to something that mlflow understands"""
    parsed = urlparse(uri)
    try:
        client.get_registered_model(parsed.hostname)
        return _parse_model_ref(parsed, client)
    except MlflowException:
        return _parse_runid_ref(parsed, client)


def _parse_model_ref(parsed: ParseResult, client: MlflowClient):
    model = parsed.hostname
    path = parsed.path.lstrip("/")
    if path.isdigit():
        mv = client.get_model_version(model, int(path))
        run = client.get_run(mv.run_id)
        return (
            "models:/{}/{}".format(model, path),
            run.data.tags,
            run.data.params,
        )
    if not path:
        stage = "none"  # TODO allow setting default stage from config
    else:
        stage = path.lower()
    results = client.get_latest_versions(model, stages=[stage])
    if not results:
        raise SpecError(
            "No versions found for model {} in stage {}".format(model, stage)
        )
    run = client.get_run(results[0].run_id)
    return (
        "models:/{}/{}".format(model, results[0].version),
        run.data.tags,
        run.data.params,
    )


def _parse_runid_ref(parsed: ParseResult, client: MlflowClient):
    runid = parsed.hostname
    run = client.get_run(runid)
    path = parsed.path.lstrip("/")
    if path:
        return (
            "runs:/{}/{}".format(runid, path),
            run.data.tags,
            run.data.params,
        )
    else:
        artifacts = client.list_artifacts(runid)
        if not artifacts:
            raise SpecError("Run {} has no artifacts".format(runid))
        elif len(artifacts) == 1:
            return (
                "runs:/{}/{}".format(runid, artifacts[0].path),
                run.data.tags,
                run.data.params,
            )
        else:
            # TODO allow setting default path from config
            raise SpecError(
                (
                    "Run {} has more than 1 artifact ({})."
                    "Please specify path like "
                    "mlflows://<runid>/path/to/artifact in "
                    "CREATE MODEL or ML_PREDICT"
                ).format(runid, [x.path for x in artifacts])
            )
