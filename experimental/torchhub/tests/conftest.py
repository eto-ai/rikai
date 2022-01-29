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
import os

import pytest
from pyspark.sql import SparkSession

from rikai.spark.utils import get_default_jar_version, init_spark_session


@pytest.fixture(scope="session")
def rikai_package_name():
    name = "ai.eto:rikai_2.12"
    scala_version = os.getenv("SCALA_VERSION")
    if scala_version and scala_version.startswith("2.13"):
        name = "ai.eto:rikai_2.13"
    return name


@pytest.fixture(scope="module")
def spark(rikai_package_name) -> SparkSession:
    rikai_version = get_default_jar_version(use_snapshot=True)

    return init_spark_session(
        dict(
            [
                (
                    "spark.jars.packages",
                    ",".join(
                        [
                            "{}:{}".format(rikai_package_name, rikai_version),
                        ]
                    ),
                ),
                (
                    "spark.rikai.sql.ml.registry.torchhub.impl",
                    "ai.eto.rikai.sql.model.torchhub.TorchHubRegistry",
                ),
            ]
        )
    )
