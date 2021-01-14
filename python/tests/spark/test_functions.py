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

"""Unit tests for Rikai provided spark UDFs."""

import os

from pyspark.sql.functions import col, lit

from rikai.spark.functions import area, image_copy
from rikai.testing.spark import SparkTestCase
from rikai.types import Box2d, Image


class SparkFunctionsTest(SparkTestCase):
    """Unit tests for spark UDFs."""

    def test_areas(self):
        """Test calculating bounding box's area."""
        df = self.spark.createDataFrame(
            [
                (Box2d(1, 2, 1.0, 1.0),),
                (Box2d(10, 12, 1.0, 5.0),),
            ],
            ["bbox"],
        )
        df = df.withColumn("area", area(col("bbox")))
        self.assertCountEqual((1.0, 5.0), df.select("area").toPandas()["area"])

    def test_image_copy(self):
        source_image = os.path.join(self.test_dir, "source_image")
        with open(source_image, "w") as fobj:
            fobj.write("abc")
        os.makedirs(os.path.join(self.test_dir, "out"))

        df = self.spark.createDataFrame(
            [(Image(source_image),)], ["image"]
        )  # type: pyspark.sql.DataFrame
        df = df.withColumn(
            "image",
            image_copy(col("image"), lit(os.path.join(self.test_dir, "out/"))),
        )
        data = df.collect()  # force lazy calculation
        out_file = os.path.join(self.test_dir, "out", "source_image")
        self.assertEqual(Image(out_file), data[0].image)

        with open(os.path.join(out_file)) as fobj:
            self.assertEqual("abc", fobj.read())
