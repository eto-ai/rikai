#  Copyright 2020 Rikai Authors
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

from pyspark.sql import Row
from pyspark.sql.functions import col

# Rikai
from rikai.spark.functions import label
from rikai.testing.spark import SparkTestCase
from rikai.vision import BBox
from rikai.types.geometry import Box3d, Point


class TypesTest(SparkTestCase):
    def test_labels(self):
        df = self.spark.createDataFrame(
            [("a",), ("b",), ("c",)],
            ["v"],
        ).withColumn("label", label("v"))

        df.write.mode("overwrite").format("rikai").save(self.test_dir)

        actual_df = self.spark.read.format("rikai").load(self.test_dir)
        self.assertCountEqual(df.collect(), actual_df.collect())

    def test_bbox(self):
        df = self.spark.createDataFrame(
            [Row(BBox(1, 2, 3, 4)), Row(BBox(23, 33, 44, 88))], ["bbox"]
        )
        df.show()
        print(df.collect())
        df.printSchema()

        df.write.mode("overwrite").format("rikai").save(self.test_dir)

        actual_df = self.spark.read.format("rikai").load(self.test_dir)
        self.assertCountEqual(df.collect(), actual_df.collect())

    def test_point(self):
        df = self.spark.createDataFrame([Row(Point(1, 2, 3)), Row(Point(2, 3, 4))])
        df.show()
        df.write.mode("overwrite").format("rikai").save(self.test_dir)

        actual_df = self.spark.read.format("rikai").load(self.test_dir)
        actual_df.show()
        self.assertCountEqual(df.collect(), actual_df.collect())

    def test_box3d(self):
        df = self.spark.createDataFrame([Row(Box3d(Point(1, 2, 3), 1, 2, 3, 2.5))])

        df.write.mode("overwrite").format("rikai").save(self.test_dir)

        actual_df = self.spark.read.format("rikai").load(self.test_dir)
        actual_df.printSchema()
        self.assertCountEqual(df.collect(), actual_df.collect())
