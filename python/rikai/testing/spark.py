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

import glob
import os
import tempfile
import unittest

from pyspark.sql import SparkSession

# Rikai
from rikai.logging import logger

__all__ = ["SparkTestCase"]


class SparkTestCase(unittest.TestCase):
    """Basic class for Running Spark Tests

    Tests can access an initialized :py:class:`SparkSession`
    via :py:attribute:`self.spark`
    """

    spark: SparkSession = None

    @classmethod
    def setUpClass(cls) -> None:
        import rikai

        jar_dir = os.path.join(os.path.dirname(rikai.__file__), "jars")
        rikai_jar = glob.glob(os.path.join(jar_dir, "*.jar"))
        if len(rikai_jar) == 0:
            raise ValueError(
                f"Rikai Jar is not found on {jar_dir}, please run 'sbt package' first"
            )
        jars = ":".join(rikai_jar)
        logger.info("loading jars for spark testing: %s", jars)
        cls.spark = (
            SparkSession.builder.appName("spark-test")
            .config("spark.jars", jars)
            .master("local[2]")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.spark is not None:
            cls.spark.stop()
            cls.spark = None

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.test_dir = self._tmpdir.name

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
