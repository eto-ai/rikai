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

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse

import yaml
from pyspark.sql import SparkSession

from rikai.io import open_uri
from rikai.logging import logger
from rikai.spark.sql.codegen.base import (
    ModelSpec,
    register_udf,
    Registry,
    udf_from_spec,
)
from rikai.spark.sql.exceptions import SpecError

__all__ = ["FileSystemRegistry"]


class FileModelSpec(ModelSpec):
    """Model Spec.

    Parameters
    ----------
    spec_uri : str or Path
        Spec file URI
    options : Dict[str, Any], optional
        Additionally options. If the same option exists in spec already,
        it will be overridden.
    validate : bool, default True.
        Validate the spec during construction. Default ``True``.
    """

    def __init__(
        self,
        spec_uri: Union[str, Path],
        options: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ):
        with open_uri(spec_uri) as fobj:
            spec = yaml.load(fobj, Loader=yaml.FullLoader)
        self.base_dir = os.path.dirname(spec_uri)
        spec.setdefault("options", {})
        if options:
            spec["options"].update(options)
        super().__init__(spec, validate=validate)

    def load_model(self):
        if self.flavor == "pytorch":
            from rikai.spark.sql.codegen.pytorch import load_model_from_uri

            return load_model_from_uri(self.model_uri)
        else:
            raise SpecError("Unsupported flavor {}".format(self.flavor))

    @property
    def model_uri(self):
        """Absolute model URI."""
        origin_uri = super().model_uri
        parsed = urlparse(origin_uri)
        if parsed.scheme or os.path.isabs(origin_uri):
            return origin_uri
        return os.path.join(self.base_dir, origin_uri)


def codegen_from_yaml(
    spark: SparkSession,
    spec_uri: str,
    name: Optional[str] = None,
    options: Optional[Dict[str, str]] = None,
) -> str:
    """Generate code from a YAML file.

    Parameters
    ----------
    spark : SparkSession
        A live spark session
    spec_uri : str
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
    spec = FileModelSpec(spec_uri, options=options)
    udf = udf_from_spec(spec)
    return register_udf(spark, udf, name)


class FileSystemRegistry(Registry):
    """FileSystem-based Model Registry"""

    def __init__(self, spark: SparkSession):
        self._spark = spark
        self._jvm = spark.sparkContext._jvm

    def __repr__(self):
        return "FileSystemRegistry"

    def resolve(self, spec):
        name = spec.getName()
        uri = spec.getUri()
        options = spec.getOptions()
        logger.info(f"Resolving model {name} from {uri}")
        if uri.endswith(".yml") or uri.endswith(".yaml"):
            func_name = codegen_from_yaml(self._spark, uri, name, options)
        else:
            raise ValueError(f"Model URI is not supported: {uri}")

        model = self._jvm.ai.eto.rikai.sql.model.SparkUDFModel(
            name, uri, func_name
        )
        # TODO: set options
        return model
