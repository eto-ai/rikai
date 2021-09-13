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

"""
ROS Standard Messages

http://docs.ros.org/en/noetic/api/std_msgs/html/index-msg.html

"""

from typing import Any, Tuple

from pyspark.sql.types import (
    DataType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    UserDefinedType,
)

__all__ = ["TimeType", "HeaderType"]


class TimeType(UserDefinedType):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def sqlType(cls) -> DataType:
        return StructType(
            [
                StructField("seconds", IntegerType(), nullable=False),
                StructField("nanoseconds", IntegerType(), nullable=False),
            ]
        )


class HeaderType(UserDefinedType):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def sqlType(cls) -> DataType:
        return StructType(
            fields=[
                StructField("seq", IntegerType(), nullable=False),
                StructField("stamp", TimeType(), nullable=False),
                StructField("frame_id", StringType(), nullable=False),
            ]
        )

    @classmethod
    def module(cls) -> str:
        return "rikai.spark.types.ros"

    @classmethod
    def scalaUDT(cls) -> str:
        return "org.apache.spark.sql.rikai.HeaderType"

    def serialize(self, obj: "Header") -> Tuple:
        return (obj.seq, obj.stamp, obj.frame_id)

    def deserialize(self, datum: Any) -> "Header":
        from rikai.types.ros.std import Header

        return Header(datum[0], datum[1], datum[2])

    def simpleString(self) -> str:
        return "HeaderType"
