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
from typing import Any, Callable, Dict, IO, Mapping, Optional, Union

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import DataType

from rikai.logging import logger
from rikai.spark.sql.exceptions import SpecError


class Registry(ABC):
    """Base class of a Model Registry"""

    @abstractmethod
    def resolve(self, uri: str, name: str, options: Dict[str, str]):
        """Resolve a model from a model URI.

        Parameters
        ----------
        uri : str
            Model URI
        name : str, optional
            Optional model name. Can be empty or None. If provided, it
            overrides the model name got from the model URI.
        options: dict
            Additional options passed to the model.
        """


class ModelSpec(ABC):
    """Base class of a Model spec"""

    @abstractmethod
    def validate(self):
        """Validate model spec

        Raises
        ------
        SpecError
            If the spec is not well-formatted.
        """

    @property
    @abstractmethod
    def version(self) -> str:
        """Returns spec version."""

    @property
    @abstractmethod
    def uri(self) -> str:
        """Return Model URI"""

    @property
    @abstractmethod
    def flavor(self) -> str:
        """Model flavor"""

    @property
    @abstractmethod
    def schema(self) -> DataType:
        """Return the output schema of the model."""

    @property
    @abstractmethod
    def options(self) -> Dict[str, str]:
        """Model options"""

    @property
    @abstractmethod
    def pre_processing(self) -> Optional[Callable]:
        """Return pre-processing transform if exists"""

    @property
    @abstractmethod
    def post_processing(self) -> Optional[Callable]:
        """Return post-processing transform if exists"""


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
