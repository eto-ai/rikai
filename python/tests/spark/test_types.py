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
from rikai.testing.spark import SparkTestCase
from rikai.types import Box3d, Box2d, Point, YouTubeVideo, VideoStream, Segment


class TypesTest(SparkTestCase):
    def _check_roundtrip(self, df):
        df.show()
        df.write.mode("overwrite").format("rikai").save(self.test_dir)
        actual_df = self.spark.read.format("rikai").load(self.test_dir)
        self.assertCountEqual(df.collect(), actual_df.collect())

    def test_bbox(self):
        df = self.spark.createDataFrame(
            [Row(Box2d(1, 2, 3, 4)), Row(Box2d(23, 33, 44, 88))], ["bbox"]
        )
        self._check_roundtrip(df)

    def test_point(self):
        df = self.spark.createDataFrame(
            [Row(Point(1, 2, 3)), Row(Point(2, 3, 4))]
        )
        self._check_roundtrip(df)

    def test_box3d(self):
        df = self.spark.createDataFrame(
            [Row(Box3d(Point(1, 2, 3), 1, 2, 3, 2.5))]
        )
        self._check_roundtrip(df)

    def test_youtubevideo(self):
        df = self.spark.createDataFrame(
            [
                Row(YouTubeVideo("video_id")),
                Row(YouTubeVideo("other_video_id")),
            ]
        )
        self._check_roundtrip(df)

    def test_videostream(self):
        df = self.spark.createDataFrame(
            [Row(VideoStream("uri1")), Row(VideoStream("uri2"))]
        )
        self._check_roundtrip(df)

    def test_segment(self):
        df = self.spark.createDataFrame(
            [Row(Segment(0, 10)), Row(Segment(15, -1))]
        )
        self._check_roundtrip(df)
