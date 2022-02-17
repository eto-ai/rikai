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
from typing import Any

from pyspark.serializers import CloudPickleSerializer
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType

from rikai.internal.reflection import find_class
from rikai.logging import logger
from rikai.spark.sql.codegen.mlflow_logger import KNOWN_FLAVORS
from rikai.spark.sql.exceptions import SpecError

__all__ = ["Registry"]

from rikai.spark.sql.model import ModelSpec

_pickler = CloudPickleSerializer()


def _identity(x):
    return x


class Registry(ABC):
    """Base class of a Model Registry"""

    @abstractmethod
    def make_model_spec(self, raw_spec: "ModelSpec"):
        """Make a ModelSpec from the raw model spec

        Parameters
        ----------
        spec : ModelSpec
        """

    def resolve(self, raw_spec: "ModelSpec"):
        """Resolve a model from the raw model spec.

        Parameters
        ----------
        spec : ModelSpec
        """
        name = raw_spec["name"]
        uri = raw_spec["uri"]
        logger.info(f"Resolving model {name} from {uri}")
        return udf_from_spec(self.make_model_spec(raw_spec))


def codegen_from_spec(spec: ModelSpec):
    """Resolve the codegen module from the model spec.

    Parameters
    ----------
    spec : ModelSpec

    Returns
    -------
    ModuleType
        The imported module for the specific flavor codegen
    """
    if spec.flavor in KNOWN_FLAVORS:
        codegen_module = f"rikai.spark.sql.codegen.{spec.flavor}"
    else:
        codegen_module = f"rikai.contrib.{spec.flavor}.codegen"

    try:
        codegen = importlib.import_module(codegen_module)
        return codegen
    except ModuleNotFoundError:
        logger.error(f"Unsupported model flavor: {spec.flavor}")
        raise


def udf_from_spec(spec: ModelSpec):
    """Return a UDF from a given ModelSpec

    Parameters
    ----------
    spec : ModelSpec
       Model spec payload

    Returns
    -------
    udt_ser_func, udf_func, udt_deser_func, returnType
        Spark UDF function name for the generated data.
    """
    if spec.version != "1.0":
        raise SpecError(
            f"Only spec version 1.0 is supported, got {spec.version}"
        )

    schema = spec.model_type.dataType()

    @udf(returnType=schema)
    def deserialize_return(data: bytes):
        return _pickler.loads(data)

    codegen = codegen_from_spec(spec)
    return (
        pickle_udt,
        codegen.generate_udf(spec),
        deserialize_return,
        schema,
    )


def command_from_spec(registry_class: str, row_spec: dict):
    cls = find_class(registry_class)
    registry = cls()
    return registry.resolve(row_spec)


@udf(returnType=BinaryType())
def pickle_udt(input):
    return _pickler.dumps(input)


def unpickle_transform(data: bytes) -> Any:
    return _pickler.loads(data)
