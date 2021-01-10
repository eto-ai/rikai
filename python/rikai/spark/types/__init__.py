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
    DataType,
    FloatType,
    IntegerType,
    ShortType,
    StringType,
    StructField,
    StructType,
    UserDefinedType,
)

# Rikai
import rikai
from rikai.convert import PortableDataType
from rikai.logging import logger
from rikai.types.geometry import PointType, Box3dType, Box2dType

__all__ = [
    "ImageType",
    "NDArrayType",
    "LabelType",
    "PointType",
    "Box3dType",
    "Box2dType",
]


class ImageType(UserDefinedType):
    """ImageType defines the Spark UserDefineType for Image type"""

    def __init__(self):
        super().__init__()
        self.codec = "png"

    def __repr__(self) -> str:
        return f"ImageType(codec={self.codec})"

    @classmethod
    def sqlType(cls) -> StructType:
        return StructType(fields=[StructField("uri", StringType(), nullable=False)])

    @classmethod
    def module(cls) -> str:
        return "rikai.spark.types"

    @classmethod
    def scalaUDT(cls) -> str:
        return "org.apache.spark.sql.rikai.ImageType"

    def serialize(self, obj: "Image"):
        """Serialize an Image to a Spark Row?"""
        return (obj.uri,)

    def deserialize(self, datum) -> "Image":
        from rikai.vision import Image  # pylint: disable=import-outside-toplevel

        return Image(datum[0])

    def simpleString(self) -> str:
        return "ImageType"


class NDArrayType(UserDefinedType):
    """User define type for an arbitrary :py:class:`numpy.ndarray`.

    This UDT serialize numpy.ndarray into bytes buffer.
    """

    def __repr__(self) -> str:
        return "np.ndarray"

    @classmethod
    def sqlType(cls) -> StructType:
        return StructType(
            fields=[
                StructField(
                    "dtype",
                    ShortType(),
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
        """Serialize an numpy.ndarra into Spark Row"""
        return (
            PortableDataType.from_numpy(obj.dtype).value,
            list(obj.shape),
            obj.tobytes(),
        )

    def deserialize(self, datum: Row) -> np.ndarray:
        pdt = PortableDataType(datum[0])

        return (
            np.frombuffer(datum[2], dtype=pdt.to_numpy())
            .reshape(datum[1])
            .view(rikai.numpy.ndarray)
        )

    def simpleString(self) -> str:
        return "ndarray"


class LabelType(UserDefinedType):
    """Label type"""

    def __repr__(self) -> str:
        return "LabelType"

    @classmethod
    def sqlType(cls) -> DataType:
        return StringType()

    @classmethod
    def module(cls) -> str:
        return "rikai.spark.types"

    @classmethod
    def scalaUDT(cls) -> str:
        return "org.apache.spark.sql.rikai.LabelType"

    def serialize(self, obj: "Label"):
        """Serialize a label into Spark String"""
        return obj.label

    def deserialize(self, datum: Row) -> "Label":
        from rikai.vision import Label

        return Label(datum)

    def simpleString(self) -> str:
        return "label"
