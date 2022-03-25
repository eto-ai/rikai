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

import importlib
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, TypeVar

from jsonschema.exceptions import ValidationError
from jsonschema.validators import validate

from rikai.internal.reflection import find_func
from rikai.logging import logger
from rikai.spark.sql.exceptions import SpecError
from rikai.spark.sql.schema import parse_schema

__all__ = ["ModelSpec", "ModelType", "AnonymousModelType"]


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
            "name": {"type": "string", "description": "Model name"},
            "schema": {"type": "string"},
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
            "transforms": {
                "type": "object",
                "properties": {
                    "pre": {"type": "string"},
                    "post": {"type": "string"},
                },
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
            f"Model spec not found for model/flavor: {model_type}/{flavor}"
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
        return AnonymousModelType(
            self._spec.get("schema", None),
            self.pre_processing,
            self.post_processing,
        )

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
        if "schema" in self._spec:
            return parse_schema(self._spec["schema"])
        return self.model_type.schema

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
        f = find_func(self._spec["transforms"]["pre"])
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
        f = find_func(self._spec["transforms"]["post"])
        return f(self.options)


class ModelType(ABC):
    """Declare a Rikai-compatible Model Type."""

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
        return parse_schema(self.schema())

    @abstractmethod
    def transform(self) -> Callable:
        """A callable to pre-process the data before calling inference.

        It will be feed into :class:`torch.data.DataLoader` or
        :class:`tensorflow.data.Dataset.map`.

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


class AnonymousModelType(ModelType):
    def __init__(
        self,
        schema: str,
        pre_processing: Optional[Callable],
        post_processing: Optional[Callable],
    ):
        self._schema = schema
        self.pre_processing = pre_processing
        self.post_processing = post_processing
        self.model: Optional[M] = None
        self.spec: Optional[ModelSpec] = None

        warnings.warn(
            "Using schema and pre_processing/post_processing explicitly"
            "is deprecated and will be removed in Rikai 0.2. "
            "Please migrate to a concrete ModelType."
        )

    def schema(self) -> str:
        return self._schema

    def load_model(self, spec: ModelSpec, **kwargs):
        self.model = spec.load_model()
        self.spec = spec

    def transform(self) -> Callable:
        """Adaptor for the pre-processing."""
        return self.pre_processing

    def predict(self, *args, **kwargs) -> Any:
        """Predict combines model inference and post-processing that
        converts inference outputs into Rikai types.

        """
        batch = self.model(*args, **kwargs)
        if self.post_processing:
            batch = self.post_processing(batch)
        return batch
