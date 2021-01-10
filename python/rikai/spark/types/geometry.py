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

"""Geometry User Defined Types in Spark
"""

from pyspark.sql import Row
from pyspark.sql.types import DoubleType, StructField, StructType, UserDefinedType
from rikai.logging import logger


class PointType(UserDefinedType):
    """Spark UDT for :py:class:`rikai.types.geometry.Point` class."""

    @classmethod
    def sqlType(cls) -> StructType:
        return StructType(
            fields=[
                StructField("x", DoubleType(), False),
                StructField("y", DoubleType(), False),
                StructField("z", DoubleType(), False),
            ]
        )

    @classmethod
    def module(cls) -> str:
        return "rikai.spark.types.geometry"

    @classmethod
    def scalaUDT(cls) -> str:
        return "org.apache.spark.sql.rikai.PointType"

    def serialize(self, obj: "Point"):
        """Serialize an numpy.ndarra into Spark Row"""
        return Row(x=obj.x, y=obj.y, z=obj.z)

    def deserialize(self, datum: Row) -> "Point":
        from rikai.types.geometry import Point

        if len(datum) < 3:
            logger.error(f"Deserialize Point: not sufficient data: {datum}")

        return Point(datum[0], datum[1], datum[2])

    def simpleString(self) -> str:
        return "PointType"
