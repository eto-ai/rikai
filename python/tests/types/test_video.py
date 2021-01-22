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

import numpy as np
import pytest
from rikai.types import (
    YouTubeVideo,
    VideoStream,
    SingleFrameSampler,
    SingleFrameGenerator,
)


def test_youtube():
    from IPython.display import YouTubeVideo as IYT

    vid = "pD1gDSao1eA"
    yt = YouTubeVideo(vid)
    assert yt.uri == "https://www.youtube.com/watch?v={0}".format(vid)
    assert yt.embed_url == "http://www.youtube.com/embed/{0}".format(vid)
    assert YouTubeVideo(vid) == YouTubeVideo(vid)
    assert YouTubeVideo(vid) != YouTubeVideo("othervid")


@pytest.mark.webtest
def test_youtube_sample_stream():
    vid = "pD1gDSao1eA"
    yt = YouTubeVideo(vid)
    v = yt.get_stream()
    assert isinstance(v, VideoStream)
    sampler = SingleFrameGenerator().get_sampler(v)
    isinstance(next(sampler.__iter__()), np.ndarray)
