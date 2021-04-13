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

from pathlib import Path
from typing import Any, Dict

import pytest
from pyspark.sql import SparkSession

from rikai.spark.sql.callback_service import CallbackService
from rikai.spark.sql.codegen.fs import Registry


class JvmModelSpec:
    def __init__(self, uri: str, name: str, options: Dict[str, Any]):
        self.uri = uri
        self.name = name
        self.options = options


def test_cb_service_find_registry(spark: SparkSession, tmp_path: Path):

    spec_file = tmp_path / "spec.yml"
    with spec_file.open("w") as fobj:
        fobj.write(
            """
version: "1.0"
schema: long
model:
    uri: abc.pt
    flavor: pytorch
    """
        )

    cb = CallbackService(spark)
    cb.resolve(
        "rikai.spark.sql.codegen.fs.FileSystemRegistry",
        JvmModelSpec(str(spec_file), "foo", {})
    )
    assert isinstance(
        cb.registry_map["rikai.spark.sql.codegen.fs.FileSystemRegistry"],
        Registry,
    )

    with pytest.raises(ModuleNotFoundError):
        cb.resolve("rikai.spark.not.exist.Registry", None)
    assert len(cb.registry_map) == 1
