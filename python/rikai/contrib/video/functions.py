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
"""
Useful functions and features for working with video data
"""

from typing import Optional, Union

# Standard library
from urllib.parse import urlparse

# Third Party
from pyspark.sql.functions import udf
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Rikai
from rikai.types.video import (
    Segment,
    SingleFrameSampler,
    VideoStream,
    YouTubeVideo,
)

__all__ = ["scene_detect", "SCENE_LIST_SCHEMA"]


SCENE_LIST_SCHEMA = ArrayType(
    StructType(
        [
            StructField(
                "start",
                StructType(
                    [
                        StructField("frame_num", IntegerType(), True),
                        StructField("frame_pos_sec", FloatType(), True),
                    ]
                ),
                True,
            ),
            StructField(
                "end",
                StructType(
                    [
                        StructField("frame_num", IntegerType(), True),
                        StructField("frame_pos_sec", FloatType(), True),
                    ]
                ),
                True,
            ),
        ]
    )
)


@udf(returnType=SCENE_LIST_SCHEMA)
def scene_detect(
    videos: Union[str, VideoStream, list], threshold=30.0, raise_on_error=True
):
    """Use `pyscenedetect <https://pyscenedetect.readthedocs.io/en/latest/>`_
    to split scenes in a video.

    Parameters
    ----------
    videos: str, VideoStream, or list<str>
        The video to split. If a list is given, assume they are segments of a
        longer video
    threshold: float, default 30.0
        Detection threshold (i.e., ContentDetector(threshold=threshold))
    raise_on_error: bool, default True
        Set to False to suppress exceptions during the detection run (note that
        import errors will still be raised)
    """
    try:
        from scenedetect.detectors import ContentDetector
        from scenedetect.scene_manager import SceneManager
        from scenedetect.video_manager import VideoManager
    except ImportError as e:
        raise ImportError(
            (
                "Please make sure you have pyscenedetect installed "
                "via `pip install rikai[video]`"
            )
        )
    video_manager = None
    try:
        if isinstance(videos, VideoStream):
            video_path = [videos.uri]
        elif isinstance(videos, list):
            video_path = videos
        else:
            video_path = [videos]

        video_manager = VideoManager(normalize_uri(video_path))
        scene_manager = SceneManager()  # TODO add StatsManager later
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        return [
            {
                "start": {
                    "frame_num": s[0].frame_num,
                    "frame_pos_sec": s[0].get_seconds(),
                },
                "end": {
                    "frame_num": s[1].frame_num,
                    "frame_pos_sec": s[1].get_seconds(),
                },
            }
            for s in scene_list
        ]
    except Exception as e:
        if raise_on_error:
            raise e
        else:
            return None
    finally:
        if video_manager is not None:
            video_manager.release()


def normalize_uri(uri: Union[str, list]):
    if isinstance(uri, str):
        parsed = urlparse(uri)
        if parsed.scheme == "s3":
            from rikai.contrib.s3 import create_signed_url

            return create_signed_url(parsed.netloc, parsed.path)
        return uri
    else:
        return [normalize_uri(u) for u in uri]
