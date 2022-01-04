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

import importlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from jsonschema import validate, ValidationError
from pyspark.serializers import CloudPickleSerializer
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType

from rikai.internal.reflection import find_class
from rikai.logging import logger
from rikai.spark.sql.codegen.mlflow_logger import KNOWN_FLAVORS
from rikai.spark.sql.exceptions import SpecError
from rikai.spark.sql.schema import parse_schema

__all__ = ["Registry", "ModelSpec"]


_pickler = CloudPickleSerializer()


def _identity(x):
    return x


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
            or self._spec["transforms"]["pre"] is None
        ):
            # Passthrough
            return _identity
        f = find_class(self._spec["transforms"]["pre"])
        return f(self.options)

    @property
    def post_processing(self) -> Optional[Callable]:
        """Return post-processing transform if exists"""
        if (
            "transforms" not in self._spec
            or "post" not in self._spec["transforms"]
            or self._spec["transforms"]["post"] is None
        ):
            # Passthrough
            return _identity
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
    udt_ser_func, udf_func, udt_deser_func, returnType
        Spark UDF function name for the generated data.
    """
    if spec.version != "1.0":
        raise SpecError(
            f"Only spec version 1.0 is supported, got {spec.version}"
        )

    if spec.flavor in KNOWN_FLAVORS:
        codegen_module = f"rikai.spark.sql.codegen.{spec.flavor}"
    else:
        codegen_module = f"rikai.contrib.{spec.flavor}.codegen"

    @udf(returnType=spec.schema)
    def deserialize_return(data: bytes):
        return _pickler.loads(data)

    try:
        codegen = importlib.import_module(codegen_module)
        return (
            pickle_udt,
            codegen.generate_udf(spec),
            deserialize_return,
            spec.schema,
        )
    except ModuleNotFoundError:
        logger.error(f"Unsupported model flavor: {spec.flavor}")
        raise


def command_from_spec(registry_class: str, row_spec: dict):
    cls = find_class(registry_class)
    registry = cls()
    return registry.resolve(row_spec)


@udf(returnType=BinaryType())
def pickle_udt(input):
    return _pickler.dumps(input)


def unpickle_transform(data: bytes) -> Any:
    return _pickler.loads(data)
