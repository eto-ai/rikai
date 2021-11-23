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

from rikai.types import SingleFrameSampler, VideoStream, YouTubeVideo


@pytest.mark.timeout(30)
@pytest.mark.webtest
def test_youtube():
    from IPython.display import YouTubeVideo as IYT

    vid = "pD1gDSao1eA"
    yt = YouTubeVideo(vid)
    assert yt.uri == "https://www.youtube.com/watch?v={0}".format(vid)
    assert yt.embed_url == "http://www.youtube.com/embed/{0}".format(vid)
    assert YouTubeVideo(vid) == YouTubeVideo(vid)
    assert YouTubeVideo(vid) != YouTubeVideo("othervid")


@pytest.mark.skip(reason="flaky due to youtube changes")
@pytest.mark.timeout(30)
@pytest.mark.webtest
def test_youtube_sample_stream():
    vid = "pD1gDSao1eA"
    yt = YouTubeVideo(vid)
    v = yt.get_stream()
    assert isinstance(v, VideoStream)
    isinstance(next(v.__iter__()), np.ndarray)


@pytest.mark.timeout(30)
@pytest.mark.webtest
def test_youtube_show():
    vid = "pD1gDSao1eA"
    yt = YouTubeVideo(vid)
    result = yt._repr_html_()
    assert result == yt.display()._repr_html_()
    # TODO actually parse the html and check kwargs
    from IPython.display import YouTubeVideo as IYT

    expected = IYT(vid)._repr_html_()
    assert result == expected


@pytest.mark.skip(reason="flaky due to youtube changes")
@pytest.mark.timeout(30)
@pytest.mark.webtest
def test_video_show():
    vid = "pD1gDSao1eA"
    v = YouTubeVideo(vid).get_stream()
    result = v._repr_html_()
    assert result == v.display()._repr_html_()
    # TODO actually parse the html and check kwargs
    from IPython.display import Video as IV

    expected = IV(v.uri)._repr_html_()
    assert result == expected
