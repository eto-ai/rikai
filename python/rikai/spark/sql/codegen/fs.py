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

from typing import Any, Dict, IO, Mapping, Optional, Union

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
        spec.setdefault("options", {})
        if options:
            spec["options"].update(options)
        super().__init__(spec, validate=validate)

    def load_model(self):
        if self.flavor == "pytorch":
            from rikai.spark.sql.codegen.pytorch import load_model_from_uri

            return load_model_from_uri(self.uri)
        else:
            raise SpecError("Unsupported flavor {}".format(self.flavor))


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
        spec = FileModelSpec(fobj, options=options)
    udf = udf_from_spec(spec)
    return register_udf(spark, udf, name)


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

        model = self._jvm.ai.eto.rikai.sql.model.SparkUDFModel(
            name, uri, func_name
        )
        # TODO: set options
        return model
