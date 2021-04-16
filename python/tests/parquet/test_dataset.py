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
import shutil

# Third Party
from pyspark.sql import Row, SparkSession

# Rikai
from rikai.parquet import Dataset
from rikai.testing.asserters import assert_count_equal


def test_select_columns(spark: SparkSession, tmp_path: Path):
    """Test reading rikai dataset with selected columns."""
    df = spark.createDataFrame(
        [
            Row(id=1, col1="value", col2=123),
            Row(id=2, col1="more", col2=456),
        ]
    )
    df.write.format("rikai").save(str(tmp_path))

    dataset = Dataset(str(tmp_path), columns=["id", "col1"])
    actual = sorted(list(dataset), key=lambda x: x["id"])

    assert_count_equal(
        [{"id": 1, "col1": "value"}, {"id": 2, "col1": "more"}], actual
    )

    shutil.rmtree(tmp_path)
