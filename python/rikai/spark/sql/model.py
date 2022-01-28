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
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional
import warnings

from jsonschema.exceptions import ValidationError
from jsonschema.validators import validate

from rikai.internal.reflection import find_func
from rikai.logging import logger
from rikai.spark.sql.exceptions import SpecError
from rikai.spark.sql.schema import parse_schema


__all__ = ["SpecPayload", "ModelType"]


# JSON schema specification for the model payload specifications
# used to validate model spec input
SPEC_PAYLOAD_SCHEMA = {
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
                "model_type": {"type": "string"},
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
    "required": ["version", "model"],
}


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
            model_mod = importlib.import_module(model_module)
            return getattr(model_mod, "MODEL_TYPE", None)
        except ModuleNotFoundError:
            pass
    else:
        raise ModuleNotFoundError(
            f"Model spec not found for model: {model_type}/{flavor}"
        )


class SpecPayload(ABC):
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

    def validate(self):
        """Validate model spec

        Raises
        ------
        SpecError
            If the spec is not well-formatted.
        """
        logger.debug("Validating spec: %s", self._spec)
        try:
            validate(instance=self._spec, schema=SPEC_PAYLOAD_SCHEMA)
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
    @abstractmethod
    def schema(self) -> str:
        pass

    def dataType(self) -> "pyspark.sql.types.DataType":
        return parse_schema(self.schema())

    @abstractmethod
    def load_model(self, raw_spec: SpecPayload, device: str = None):
        pass

    @abstractmethod
    def transform(self) -> Callable:
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        pass

    def __call__(self, *args, **kwargs) -> Any:
        return self.predict(*args, **kwargs)

    def release(self):
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

        warnings.warn(
            "Using schema and pre_processing/post_processing explicitly"
            "is deprecated. Please migrate to an concrete ModelType")

    def schema(self) -> str:
        return self._schema

    def load_model(self, raw_spec: SpecPayload, device: str = None):
        # TODO: This interface is too tight to Pytorch
        self.model = raw_spec.load_model()
        self.model.to(device)
        self.model.eval()

    def transform(self) -> Callable:
        return self.pre_processing

    def predict(self, *args, **kwargs) -> Any:
        batch = self.model(*args, **kwargs)
        if self.post_processing:
            batch = self.post_processing(batch)
        return batch
