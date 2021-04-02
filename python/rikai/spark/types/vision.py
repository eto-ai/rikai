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


from pyspark.sql.types import (
    BinaryType,
    StringType,
    StructField,
    StructType,
    UserDefinedType,
)

__all__ = ["ImageType"]


class ImageType(UserDefinedType):
    """ImageType defines the Spark UserDefineType for Image type"""

    def __init__(self):
        super().__init__()
        self.codec = "png"

    def __repr__(self) -> str:
        return f"ImageType(codec={self.codec})"

    @classmethod
    def sqlType(cls) -> StructType:
        return StructType(
            fields=[
                StructField("data", BinaryType(), nullable=True),
                StructField("uri", StringType(), nullable=True),
            ]
        )

    @classmethod
    def module(cls) -> str:
        return "rikai.spark.types.vision"

    @classmethod
    def scalaUDT(cls) -> str:
        return "org.apache.spark.sql.rikai.ImageType"

    def serialize(self, obj: "Image"):
        """Serialize an Image to a Spark Row?"""
        return (obj.data, obj.uri)

    def deserialize(self, datum) -> "Image":
        from rikai.types.vision import Image

        return Image(datum[0] or datum[1])

    def simpleString(self) -> str:
        return "ImageType"
