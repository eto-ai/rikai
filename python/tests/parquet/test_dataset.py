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

# Third Party
from pyspark.sql import Row

# Rikai
from rikai.parquet import Dataset
from rikai.testing import SparkTestCase


class DatasetTest(SparkTestCase):
    def test_select_columns(self):
        """Test reading rikai dataset with selected columns."""
        df = self.spark.createDataFrame(
            [Row(id=1, col1="value", col2=123), Row(id=2, col1="more", col2=456)]
        )
        df.write.format("rikai").save(self.test_dir)

        dataset = Dataset(self.test_dir, columns=["id", "col1"])
        actual = []
        for example in dataset:
            actual.append(example)
        actual = sorted(actual, key=lambda x: x["id"])

        self.assertCountEqual(
            [{"id": 1, "col1": "value"}, {"id": 2, "col1": "more"}], actual
        )
