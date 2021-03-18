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
from typing import Dict, Optional

import yaml
from jsonschema import validate
from pyspark.sql import SparkSession

from rikai.io import open_uri
from rikai.logging import logger
from rikai.spark.sql.codegen.base import Registry

# YAML-Spec SCHEMA
SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {"type": "string"},
        "name": {"type": "string"},
        "schema": {"type": "string"},
        "model": {
            "type": "object",
            "description": "model description",
            "properties": {
                "uri": "string",
                "flavor": "string",
            },
            "required": ["uri"],
        },
        "options": {"type": "object"},
    },
    "required": ["version", "schema", "model"],
}


def codegen_from_yaml(
    spark: SparkSession,
    uri: str,
    name: Optional[str] = None,
    options: Dict[str, str] = {},
):
    spec = yaml.load(open_uri(uri), Loader=yaml.FullLoader)
    print(spec)
    validate(instance=spec, schema=SPEC_SCHEMA)

    func_name = f"{name}_{secrets.token_hex(4)}"
    logger.info(f"Creating pandas_udf with name {func_name}")
    return func_name


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

        model = self._jvm.ai.eto.rikai.sql.model.fs.FileSystemModel(
            name, uri, func_name
        )
        # TODO: set options
        return model
