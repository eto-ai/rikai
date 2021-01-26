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

# Standard
from pathlib import Path

# Third Party
import pytest
from pyspark.sql import SparkSession

# Rikai
import rikai


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    jar_dir = Path(rikai.__file__).parent / "jars"
    jars = [
        (jar_dir / jar_file).as_posix()
        for jar_file in jar_dir.iterdir()
        if jar_file.suffix == ".jar"
    ]
    return (
        SparkSession.builder.appName("spark-test")
        .config("spark.jars", ":".join(jars))
        .master("local[2]")
        .getOrCreate()
    )
