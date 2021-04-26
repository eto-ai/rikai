#  Copyright (c) 2021 Rikai Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest
from pyspark.sql import Row, SparkSession

from rikai.contrib.video.functions import scene_detect

# Rikai
from rikai.types import Box2d, Image, Segment, VideoStream


@pytest.mark.timeout(10)
@pytest.mark.webtest
def test_scene_detect(spark: SparkSession, asset_path: Path):
    video = VideoStream(str(asset_path / "big_buck_bunny_short.mp4"))
    df = spark.createDataFrame([(video,)], ["video"])
    result = [
        r.asDict(True)
        for r in df.withColumn("scenes", scene_detect("video")).first()[
            "scenes"
        ]
    ]
    expected = [
        {
            "start": {"frame_num": 0, "frame_pos_sec": 0.0},
            "end": {"frame_num": 300, "frame_pos_sec": 10.010000228881836},
        }
    ]
    for rs, xp in zip(result, expected):
        pdt.assert_frame_equal(pd.DataFrame(rs), pd.DataFrame(xp))
