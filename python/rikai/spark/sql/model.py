#  Copyright 2022 Rikai Authors
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

import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, TypeVar

from jsonschema.exceptions import ValidationError
from jsonschema.validators import validate

from rikai.internal.reflection import find_func
from rikai.logging import logger
from rikai.spark.sql.exceptions import SpecError
from rikai.spark.sql.schema import parse_schema

__all__ = ["ModelSpec", "ModelType"]


M = TypeVar("M")  # Model Type


# JSON schema specification for the model payload specifications
# used to validate model spec input
def gen_schema_spec(required_cols):
    return {
        "type": "object",
        "properties": {
            "version": {
                "type": "string",
                "description": "Model SPEC format version",
            },
            "schema": {"type": "string"},
            "name": {"type": "string", "description": "Model name"},
            "model": {
                "type": "object",
                "description": "model description",
                "properties": {
                    "uri": {"type": "string"},
                    "flavor": {"type": "string"},
                    "type": {"type": "string"},
                },
                "required": required_cols,
            },
        },
        "required": ["version", "model"],
    }


SPEC_PAYLOAD_SCHEMA = gen_schema_spec(["uri"])
NOURI_SPEC_SCHEMA = gen_schema_spec(["flavor", "type"])


def _identity(x):
    return x


def is_fully_qualified_name(name: str) -> bool:
    return "." in name


def parse_model_type(flavor: str, model_type: str):
    model_modules_candidates = []

    if is_fully_qualified_name(model_type):
        model_modules_candidates.append(model_type)
    else:
        # Try registered models first
        try:
            registered_models = find_func(f"rikai.{flavor}.models.MODEL_TYPES")
            if registered_models:
                return registered_models[model_type]
        except (ModuleNotFoundError, KeyError):
            pass
        model_modules_candidates.extend(
            [
                f"rikai.{flavor}.models.{model_type}",
                f"rikai.contrib.{flavor}.models.{model_type}",
            ]
        )
    for model_module in model_modules_candidates:
        try:
            return find_func(f"{model_module}.MODEL_TYPE")
        except ModuleNotFoundError:
            pass
    else:
        raise ModuleNotFoundError(
            f"Model type not found for model/flavor: {model_type}/{flavor}"
        )


class ModelSpec(ABC):
    """Model Spec Payload

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

    def validate(self, schema=SPEC_PAYLOAD_SCHEMA):
        """Validate model spec

        Raises
        ------
        SpecError
            If the spec is not well-formatted.
        """
        logger.debug("Validating spec: %s", self._spec)
        try:
            validate(instance=self._spec, schema=schema)
        except ValidationError as e:
            raise SpecError(e.message) from e
        if not self.flavor or not self.model_type:
            raise SpecError("Missing model flavor or model type")

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

    @property
    def model_type(self) -> "ModelType":
        """Return model type"""
        mtype = self._spec["model"].get("type", None)
        if mtype:
            return parse_model_type(self.flavor, mtype)

    @abstractmethod
    def load_model(self) -> Any:
        """Load the model artifact specified in this spec"""

    def load_label_fn(self) -> Optional[Callable]:
        """Load the function that maps label id to human-readable string"""
        if "labels" in self._spec:
            uri = self._spec["labels"].get("uri")
            if uri:
                with open(uri) as fh:
                    dd = json.load(fh)
                return lambda label_id: dd[label_id]
            func = self._spec["labels"].get("func")
            if func:
                return find_func(func)
        return None

    @property
    def flavor(self) -> str:
        """Model flavor"""
        return self._spec["model"].get("flavor", "")

    @property
    def schema(self) -> str:
        """Return the output schema of the model."""
        if "schema" in self._spec:
            return parse_schema(self._spec["schema"])
        return parse_schema(self.model_type.schema())

    @property
    def options(self) -> Dict[str, Any]:
        """Model options"""
        return self._spec["options"]


class ModelType(ABC):
    """Base-class for a Model Type.

    A Model Type defines the functionalities which is required to run
    an arbitrary ML models in SQL ML, including:

    - Result schema: :py:meth:`schema`.
    - :py:meth:`transform`, pre-processing routine.
    - :py:meth:`predict`, inference **AND** post-processing routine.
    """

    @abstractmethod
    def load_model(self, spec: ModelSpec, **kwargs):
        """Lazy loading the model from a :class:`ModelSpec`."""
        pass

    @abstractmethod
    def schema(self) -> str:
        """Return the string value of model schema.

        Examples
        --------

        >>> model_type.schema()
        ... "array<struct<box:box2d, score:float, label_id:int>>"
        """
        pass

    def dataType(self) -> "pyspark.sql.types.DataType":
        """Returns schema as :py:class:`pyspark.sql.types.DataType`."""
        return parse_schema(self.schema())

    @abstractmethod
    def transform(self) -> Callable:
        """A callable to pre-process the data before calling inference.

        It will be feed into :py:class:`torch.data.DataLoader` or
        :py:meth:`tensorflow.data.Dataset.map`.

        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Run model inference and convert return types into
        Rikai-compatible types.

        """
        pass

    def __call__(self, *args, **kwargs) -> Any:
        return self.predict(*args, **kwargs)

    def release(self):
        """Release underneath resources if applicable.

        It will be called after a model runner finishes a partition in Spark.
        """
        # Noop by default
        pass
