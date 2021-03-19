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
from typing import Any, Dict, Optional, Union, IO, Mapping

import yaml
from jsonschema import ValidationError, validate
from pyspark.sql import SparkSession

from rikai.io import open_uri
from rikai.logging import logger
from rikai.spark.sql.codegen.base import Registry
from rikai.spark.sql.exceptions import SpecError
from rikai.spark.sql.schema import parse_schema


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
    },
    "required": ["version", "schema", "model"],
}


class ModelSpec:
    """Model Spec"""

    def __init__(
        self,
        spec: Union[bytes, str, IO, Dict[str, Any]],
        validate: bool = True,
    ):
        if not isinstance(spec, Mapping):
            spec = yaml.load(spec, Loader=yaml.FullLoader)
        self._spec = spec

        if validate:
            self.validate()

    def validate(self):
        """Validate model spec

        Raises
        ------
        SpecError
            If the spec is not well-formatted.
        """
        logger.debug("Validate spec: %s", self._spec)
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
        return self._spec.get("options", {})


def codegen_from_yaml(
    spark: SparkSession,
    uri: str,
    name: Optional[str] = None,
    options: Dict[str, str] = {},
):
    with open_uri(uri) as fobj:
        spec = ModelSpec(fobj)

    if spec.version != 1.0:
        raise SpecError(
            f"Only spec version 1.0 is supported, got {spec.version}"
        )

    if spec.flavor == "pytorch":
        from rikai.spark.sql.codegen.pytorch import generate_udf

        udf = generate_udf(spec.uri, spec.schema, spec.options)
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
