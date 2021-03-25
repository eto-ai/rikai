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

# Third Party
import pytest
from torch.utils.data import DataLoader  # Prevent DataLoader hangs
from pyspark.sql import SparkSession

from rikai.spark.sql import init


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    session = (
        SparkSession.builder.appName("spark-test")
        .config("spark.jars.packages", "ai.eto:rikai_2.12:0.0.3-SNAPSHOT")
        .config(
            "spark.sql.extensions",
            "ai.eto.rikai.sql.spark.RikaiSparkSessionExtensions",
        )
        .config(
            "rikai.sql.ml.registry.test.impl",
            "ai.eto.rikai.sql.model.testing.TestRegistry",
        )
        .config(
            "rikai.sql.ml.registry.file.impl",
            "ai.eto.rikai.sql.model.fs.FileSystemRegistry",
        )
        .config(
            "spark.driver.extraJavaOptions",
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
        .master("local[2]")
        .getOrCreate()
    )
    init(session)
    return session


@pytest.fixture
def asset_path() -> Path:
    return Path(__file__).parent / "assets"
