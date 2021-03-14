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

from typing import Dict, Optional, Union
from pathlib import Path

import yaml
from pyspark.sql import SparkSession

from rikai.io import open_uri
from rikai.logging import logger
from rikai.spark.sql.schema import parse_schema
from rikai.spark.sql.codegen.exceptions import ModelSpecFormatError
from rikai.spark.sql.codegen.runners import pytorch_runner


def codegen_from_yaml(
    spark: SparkSession, yaml_uri: str, name: Optional[str] = None
):
    # TODO(lei): find a schema validation for spec?
    spec = yaml.load(open_uri(yaml_uri), Loader=yaml.FullLoader)  # type: Dict
    print(spec)
    version = spec.get("version", None)
    if version is None:
        raise ModelSpecFormatError("Missing version field in YAML spec")
    name = spec.get("name", name) if not name else name
    if name is None:
        raise ModelSpecFormatError("Missing model name")
    try:
        model = spec["model"]
    except KeyError as e:
        raise ModelSpecFormatError(
            f"Model section is missing from YAML spec: {yaml_uri}"
        ) from e
    try:
        model_uri = model["uri"]
    except KeyError as e:
        raise ModelSpecFormatError(
            f"Miss model.uri from YAML spec: {yaml_uri}"
        ) from e
    if model_uri is None:
        raise ModelSpecFormatError(
            f"Miss model.uri from YAML spec: {yaml_uri}"
        )

    try:
        schema_str = spec["schema"]
        schema = parse_schema(schema_str)  # parse_schema(schema_str)
    except KeyError:
        raise ModelSpecFormatError("Missing schema from YAML spec")

    func_name = name
    flavor = model["flavor"]
    if flavor == "pytorch":
        print("REGISTER WITH SCHEMA: ", schema)
        spark.udf.register(
            func_name, pytorch_runner(yaml_uri, model_uri, schema)
        )
    else:
        raise ModelSpecFormatError(f"Unsupported model flavor: ${flavor}")
    return func_name


class FileSystemModel:
    """Filesystem-based models."""

    def __init__(
        self,
        uri: Union[str, Path],
        name: Optional[str] = None,
        options: Optional[Dict[str, str]] = None,
    ):
        self.uri = str(uri)
        self.name = name
        self.options = options

    def codegen(self, spark: SparkSession):
        """Code Generation for a Filesystem based model.

        Parameters
        ----------
        spark : SparkSession
            SparkSession

        Returns
        -------

        """
        if self.uri.endswith(".yml") or self.uri.endswith(".yaml"):
            logger.info(f"Load model from YAML file: {self.uri}")
            func_name = codegen_from_yaml(spark, self.uri)
        else:
            raise ValueError(f"Not supported model URI: ${self.uri}")
        print(f"Code generation for model_name={self.name}")
        return (
            spark.sparkContext._jvm.ai.eto.rikai.sql.model.fs.FileSystemModel(
                self.uri, self.name, None
            )
        )
