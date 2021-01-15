#  Copyright 2021 Rikai Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from pyspark.sql.types import (
    StringType,
    StructField,
    StructType,
    UserDefinedType,
)

__all__ = ["YouTubeVideoType", "VideoStreamType"]


class VideoStreamType(UserDefinedType):
    """VideoStreamType defines the Spark UserDefineType for
    a given video stream
    """

    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return f"VideoType"

    @classmethod
    def sqlType(cls) -> StructType:
        return StructType(fields=[StructField("uri", StringType(), nullable=False)])

    @classmethod
    def module(cls) -> str:
        return "rikai.spark.types"

    @classmethod
    def scalaUDT(cls) -> str:
        return "org.apache.spark.sql.rikai.VideoStreamType"

    def serialize(self, obj: "VideoStream"):
        """Serialize a VideoStream to a Spark Row"""
        return (obj.uri,)

    def deserialize(self, datum) -> "VideoStream":
        from rikai.types import VideoStream  # pylint: disable=import-outside-toplevel

        return VideoStream(datum[0])

    def simpleString(self) -> str:
        return "VideoStreamType"


class YouTubeVideoType(UserDefinedType):
    """YouTubeVideoType defines the Spark UserDefineType for
    a piece of YouTube video content (i.e., corresponds to a given
    youtube id but can have multiple streams)
    """

    def __init__(self):
        super().__init__()

    def __repr__(self) -> str:
        return "YouTubeVideoType"

    @classmethod
    def sqlType(cls) -> StructType:
        return StructType(fields=[StructField("vid", StringType(), nullable=False)])

    @classmethod
    def module(cls) -> str:
        return "rikai.spark.types"

    @classmethod
    def scalaUDT(cls) -> str:
        return "org.apache.spark.sql.rikai.YouTubeVideoType"

    def serialize(self, obj: "YouTubeVideo"):
        """Serialize a YouTubeVideo"""
        return (obj.vid,)

    def deserialize(self, datum) -> "YouTubeVideo":
        from rikai.types import YouTubeVideo  # pylint: disable=import-outside-toplevel

        return YouTubeVideo(datum[0])

    def simpleString(self) -> str:
        return "YouTubeVideoType"
