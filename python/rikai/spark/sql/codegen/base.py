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
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from jsonschema import validate, ValidationError
from pyspark.sql import SparkSession

from rikai.internal.reflection import find_class
from rikai.logging import logger
from rikai.spark.sql.exceptions import SpecError
from rikai.spark.sql.schema import parse_schema

__all__ = ["Registry", "ModelSpec"]


class Registry(ABC):
    """Base class of a Model Registry"""

    @abstractmethod
    def resolve(self, spec: "ModelSpec"):
        """Resolve a model from a model URI.

        Parameters
        ----------
        spec : ModelSpec
        """


# JSON schema specification for the model specifications
# used to validate model spec input
MODEL_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {
            "type": "string",
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


class ModelSpec(ABC):
    """Model Spec.

    Parameters
    ----------
    spec : dict
        Dictionary representation of an input spec
    validate : bool, default True.
        Validate the spec during construction. Default ``True``.
    """

    def __init__(self, spec: Dict[str, Any], validate: bool = True):
        self._spec = spec
        self._spec["options"] = self._spec.get("options", {})
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
            validate(instance=self._spec, schema=MODEL_SPEC_SCHEMA)
        except ValidationError as e:
            raise SpecError(e.message) from e

    @property
    def version(self) -> str:
        """Returns spec version."""
        return str(self._spec["version"])

    @property
    def name(self) -> str:
        """Return model name"""
        return self._spec["name"]

    @property
    def model_uri(self) -> str:
        """Return Model artifact URI"""
        return self._spec["model"]["uri"]

    @abstractmethod
    def load_model(self) -> Any:
        """Load the model artifact specified in this spec"""

    @property
    def flavor(self) -> str:
        """Model flavor"""
        return self._spec["model"].get("flavor", "")

    @property
    def schema(self) -> str:
        """Return the output schema of the model."""
        return parse_schema(self._spec["schema"])

    @property
    def options(self) -> Dict[str, Any]:
        """Model options"""
        return self._spec["options"]

    @property
    def pre_processing(self) -> Optional[Callable]:
        """Return pre-processing transform if exists"""
        if (
            "transforms" not in self._spec
            or "pre" not in self._spec["transforms"]
        ):
            # Passthrough
            return lambda x: x
        f = find_class(self._spec["transforms"]["pre"])
        return f(self.options)

    @property
    def post_processing(self) -> Optional[Callable]:
        """Return post-processing transform if exists"""
        if (
            "transforms" not in self._spec
            or "post" not in self._spec["transforms"]
        ):
            # Passthrough
            return lambda x: x
        f = find_class(self._spec["transforms"]["post"])
        return f(self.options)


def udf_from_spec(spec: ModelSpec):
    """Return a UDF from a given ModelSpec

    Parameters
    ----------
    spec : ModelSpec
        A model spec

    Returns
    -------
    str
        Spark UDF function name for the generated data.
    """
    if spec.version != "1.0":
        raise SpecError(
            f"Only spec version 1.0 is supported, got {spec.version}"
        )

    if spec.flavor == "pytorch":
        from rikai.spark.sql.codegen.pytorch import generate_udf

        return generate_udf(spec)
    else:
        raise SpecError(f"Unsupported model flavor: {spec.flavor}")


def register_udf(spark: SparkSession, udf: Callable, name: str) -> str:
    """
    Register a given UDF with the give Spark session under the given name.
    """
    func_name = f"{name}_{secrets.token_hex(4)}"
    spark.udf.register(func_name, udf)
    logger.info(f"Created model inference pandas_udf with name {func_name}")
    return func_name
