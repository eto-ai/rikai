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

import pytest
from pyspark.sql import SparkSession

from rikai.spark.utils import get_default_jar_version, init_spark_session


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    rikai_version = get_default_jar_version(use_snapshot=True)

    return init_spark_session(
        dict(
            [
                (
                    "spark.jars.packages",
                    ",".join(
                        [
                            "ai.eto:rikai_2.12:{}".format(rikai_version),
                        ]
                    ),
                ),
            ]
        )
    )
