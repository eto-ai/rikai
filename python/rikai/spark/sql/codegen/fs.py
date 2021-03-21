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

import secrets
from typing import Callable, IO, Any, Dict, Mapping, Optional, Union

import yaml
from jsonschema import ValidationError, validate
from pyspark.sql import SparkSession

from rikai.io import open_uri
from rikai.logging import logger
from rikai.spark.sql.codegen.base import Registry
from rikai.spark.sql.exceptions import SpecError
from rikai.spark.sql.schema import parse_schema
from rikai.internal.reflection import find_class

__all__ = ["FileSystemRegistry"]

# YAML-Spec SCHEMA
SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {
            "type": "number",
            "description": "Model SPEC format version",
        },
        "name": {"type": "string", "description": "Model name"},
        "schema": {"type": "string"},
        "model": {
            "type": "object",
            "description": "model description",
            "properties": {
                "uri": {"type": "string"},
                "flavor": {"type": "string"},
            },
            "required": ["uri"],
        },
        "transforms": {
            "type": "object",
            "properties": {
                "pre": {"type": "string"},
                "post": {"type": "string"},
            },
        },
    },
    "required": ["version", "schema", "model"],
}


class ModelSpec:
    """Model Spec.

    Parameters
    ----------
    spec : str, bytes, dict or file object
        String content of the serialized spec, or a dict
    options : Dict[str, Any], optional
        Additionally options. If the same option exists in spec already,
        it will be overridden.
    validate : bool, default True.
        Validate the spec during construction. Default ``True``.
    """

    def __init__(
        self,
        spec: Union[bytes, str, IO, Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ):
        if not isinstance(spec, Mapping):
            spec = yaml.load(spec, Loader=yaml.FullLoader)
        self._spec = spec
        self._options: Dict[str, Any] = self._spec.get("options", {})
        if options:
            self._options.update(options)

        if validate:
            self.validate()

    def validate(self):
        """Validate model spec

        Raises
        ------
        SpecError
            If the spec is not well-formatted.
        """
        logger.debug("Validating spec: %s", self._spec)
        try:
            validate(instance=self._spec, schema=SPEC_SCHEMA)
        except ValidationError as e:
            raise SpecError(e.message) from e

    @property
    def version(self):
        """Returns spec version."""
        return self._spec["version"]

    @property
    def uri(self):
        """Return Model URI"""
        return self._spec["model"]["uri"]

    @property
    def flavor(self):
        """Model flavor"""
        return self._spec["model"].get("flavor", "")

    @property
    def schema(self):
        """Return the output schema of the model."""
        return parse_schema(self._spec["schema"])

    @property
    def options(self):
        """Model options"""
        return self._options

    @property
    def pre_processing(self) -> Optional[Callable]:
        """Return pre-processing transform if exists"""
        if (
            "transforms" not in self._spec
            or "pre" not in self._spec["transforms"]
        ):
            return None
        f = find_class(self._spec["transforms"]["pre"])
        return f

    @property
    def post_processing(self) -> Optional[Callable]:
        """Return post-processing transform if exists"""
        if (
            "transforms" not in self._spec
            or "post" not in self._spec["transforms"]
        ):
            return None
        f = find_class(self._spec["transforms"]["post"])
        return f


def codegen_from_yaml(
    spark: SparkSession,
    uri: str,
    name: Optional[str] = None,
    options: Optional[Dict[str, str]] = None,
) -> str:
    """Generate code from a YAML file.

    Parameters
    ----------
    spark : SparkSession
        A live spark session
    uri : str
        the model spec URI
    name : model name
        The name of the model.
    options : dict
        Optional parameters passed to the model.

    Returns
    -------
    str
        Spark UDF function name for the generated data.
    """
    with open_uri(uri) as fobj:
        spec = ModelSpec(fobj, options=options)

    if spec.version != 1.0:
        raise SpecError(
            f"Only spec version 1.0 is supported, got {spec.version}"
        )

    if spec.flavor == "pytorch":
        from rikai.spark.sql.codegen.pytorch import generate_udf

        udf = generate_udf(
            spec.uri,
            spec.schema,
            spec.options,
            pre_processing=spec.pre_processing,
            post_processing=spec.post_processing,
        )
    else:
        raise SpecError(f"Unsupported model flavor: {spec.flavor}")

    func_name = f"{name}_{secrets.token_hex(4)}"
    spark.udf.register(func_name, udf)
    logger.info(f"Created model inference pandas_udf with name {func_name}")
    return func_name


class FileSystemRegistry(Registry):
    """FileSystem-based Model Registry"""

    def __init__(self, spark: SparkSession):
        self._spark = spark
        self._jvm = spark.sparkContext._jvm

    def __repr__(self):
        return "FileSystemRegistry"

    def resolve(self, uri: str, name: str, options: Dict[str, str]):
        logger.info(f"Resolving model {name} from {uri}")

        if uri.endswith(".yml") or uri.endswith(".yaml"):
            func_name = codegen_from_yaml(self._spark, uri, name, options)
        else:
            raise ValueError(f"Model URI is not supported: {uri}")

        model = self._jvm.ai.eto.rikai.sql.model.fs.FileSystemModel(
            name, uri, func_name
        )
        # TODO: set options
        return model
