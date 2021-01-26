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

"""Geometry User Defined Types in Spark
"""

from __future__ import annotations

# Third-Party
from pyspark.sql import Row
from pyspark.sql.types import (
    DoubleType,
    StructField,
    StructType,
    UserDefinedType,
)

# Rikai
from rikai.logging import logger

__all__ = ["PointType", "Box3dType", "Box2dType"]


class Box2dType(UserDefinedType):
    """User defined type for the 2D bounding box."""

    @classmethod
    def sqlType(cls) -> StructType:
        return StructType(
            fields=[
                StructField("xmin", DoubleType(), False),
                StructField("ymin", DoubleType(), False),
                StructField("xmax", DoubleType(), False),
                StructField("ymax", DoubleType(), False),
            ]
        )

    @classmethod
    def module(cls) -> str:
        return "rikai.spark.types.geometry"

    @classmethod
    def scalaUDT(cls) -> str:
        return "org.apache.spark.sql.rikai.Box2dType"

    def serialize(self, obj: "rikai.types.geometry.Box2d"):
        """Serialize a Box2d into a PySpark Row"""
        return (
            obj.xmin,
            obj.ymin,
            obj.xmax,
            obj.ymax,
        )

    def deserialize(self, datum: Row) -> "rikai.types.geometry.Box2d":
        from rikai.types.geometry import Box2d

        if len(datum) < 4:
            logger.error(f"Deserialize box2d: not sufficient data: {datum}")

        return Box2d(*datum[:4])

    def simpleString(self) -> str:
        return "box2d"


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
        """Serialize an :py:class:`PointType` into Spark Row"""
        return Row(x=obj.x, y=obj.y, z=obj.z)

    def deserialize(self, datum: Row) -> "Point":
        from rikai.types.geometry import Point

        if len(datum) < 3:
            logger.error(f"Deserialize Point: not sufficient data: {datum}")

        return Point(datum[0], datum[1], datum[2])

    def simpleString(self) -> str:
        return "PointType"


class Box3dType(UserDefinedType):
    """Spark UDT for :py:class:`~rikai.types.geometry.Box3d` class."""

    @classmethod
    def sqlType(cls) -> StructType:
        return StructType(
            fields=[
                StructField("center", PointType.sqlType(), False),
                StructField("length", DoubleType(), False),
                StructField("width", DoubleType(), False),
                StructField("height", DoubleType(), False),
                StructField("heading", DoubleType(), False),
            ]
        )

    @classmethod
    def module(cls) -> str:
        return "rikai.spark.types.geometry"

    @classmethod
    def scalaUDT(cls) -> str:
        return "org.apache.spark.sql.rikai.Box3dType"

    def serialize(self, obj: "Box3d"):
        """Serialize an Box3d into a Spark Row"""
        return Row(obj.center, obj.length, obj.width, obj.height, obj.heading)

    def deserialize(self, datum: Row) -> "Box3d":
        from rikai.types.geometry import Box3d

        if len(datum) < 5:
            logger.error(f"Deserialize Box3d: not sufficient data: {datum}")
        return Box3d(datum[0], datum[1], datum[2], datum[3], datum[4])

    def simpleString(self) -> str:
        return "Box3dType"
