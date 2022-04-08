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

"""Spark User Defined Types
"""

# Third Party
import numpy as np
from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    UserDefinedType,
)

# Rikai
import rikai
from rikai.spark.types.geometry import (
    Box2dType,
    Box3dType,
    MaskType,
    PointType,
)
from rikai.spark.types.video import (
    SegmentType,
    VideoStreamType,
    YouTubeVideoType,
)
from rikai.spark.types.vision import ImageType

__all__ = [
    "ImageType",
    "NDArrayType",
    "PointType",
    "Box3dType",
    "Box2dType",
    "VideoStreamType",
    "YouTubeVideoType",
    "MaskType",
    "SegmentType",
]


class NDArrayType(UserDefinedType):
    """User define type for an arbitrary :py:class:`numpy.ndarray`.

    This UDT serializes :py:class:`numpy.ndarray` into bytes buffer.
    """

    def __repr__(self) -> str:
        return "np.ndarray"

    @classmethod
    def sqlType(cls) -> StructType:
        return StructType(
            fields=[
                # dtype field will use dictionary encoding.
                StructField(
                    "dtype",
                    StringType(),
                    False,
                ),
                StructField(
                    "shape",
                    ArrayType(IntegerType(), False),
                    False,
                ),
                StructField(
                    "data",
                    BinaryType(),
                    False,
                ),
            ]
        )

    @classmethod
    def module(cls) -> str:
        return "rikai.spark.types"

    @classmethod
    def scalaUDT(cls) -> str:
        return "org.apache.spark.sql.rikai.NDArrayType"

    def serialize(self, obj: np.ndarray):
        """Serialize an :py:class:`numpy.ndarray` into Spark Row"""
        return (
            str(obj.dtype),
            list(obj.shape),
            obj.tobytes(),
        )

    def deserialize(self, datum: Row) -> np.ndarray:
        import rikai.numpy

        return (
            np.frombuffer(datum[2], dtype=np.dtype(datum[0]))
            .reshape(datum[1])
            .view(rikai.numpy.ndarray)
        )

    def simpleString(self) -> str:
        return "ndarray"
