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

"""Video related types and utils"""
from abc import ABC, abstractmethod
import cv2
import pafy
import pathlib
import tempfile
from urllib.parse import urlparse
import youtube_dl

from rikai.spark.types import YouTubeVideoType, VideoStreamType, SegmentType

__all__ = [
    "YouTubeVideo",
    "VideoStream",
    "VideoSampler",
    "SamplerGenerator",
    "SingleFrameSampler",
    "SingleFrameGenerator",
    "Segment",
]


class YouTubeVideo:
    """
    Represents a YouTubeVideo, the basis of many open-source video data sets.
    This classes uses the ipython display library to integrate with jupyter
    notebook display and uses youtube-dl to download and create a
    VideoStream instance which represents a particular video stream file obj
    """

    __UDT__ = YouTubeVideoType()

    def __init__(self, vid: str):
        """

        Parameters
        ----------
        vid: str
            The youtube video id
        """
        self.vid = vid
        self.uri = "https://www.youtube.com/watch?v={0}".format(self.vid)
        self.embed_url = "http://www.youtube.com/embed/{0}".format(self.vid)

    def __repr__(self) -> str:
        return "YouTubeVideo({0})".format(self.vid)

    def _repr_html_(self):
        from IPython.lib.display import YouTubeVideo

        return YouTubeVideo(self.vid)._repr_html_()

    def __eq__(self, other) -> bool:
        return isinstance(other, YouTubeVideo) and self.vid == other.vid

    def get_stream(self, download: bool = False, **kwargs) -> "VideoStream":
        """
        Get one specific stream of this youtube video

        Parameters
        ----------
        download: bool, default False
            By default just return a remote https reference. If True then
            actually download the data
        kwargs

        Returns
        -------
        v: VideoStream
            The downloaded video
        """
        # TODO add options to change extension and quality
        if download:
            with tempfile.TemporaryDirectory() as tmpdir:
                dl_path = pathlib.Path(tmpdir) / "{}.mp4".format(self.vid)
                opts = {"format": "worst[ext=mp4]", "outtmpl": dl_path}
                opts.update(kwargs)
                with youtube_dl.YoutubeDL(opts) as dl:
                    dl.download([self.uri])
                return VideoStream(dl_path)
                # TODO add option to cp to s3
        else:
            return VideoStream(getworst(pafy.new(self.uri), preftype="mp4").url)


# Pafy hasn't had a new release with getworst yet
def getworst(v_pafy, preftype="any", ftypestrict=True, vidonly=False):
    """
    Return the highest resolution video available.

    Select from video-only streams if vidonly is True
    """
    streams = v_pafy.videostreams if vidonly else v_pafy.streams

    if not streams:
        return None

    def _sortkey(x, key3d=0, keyres=0, keyftype=0):
        """ sort function for max(). """
        key3d = "3D" not in x.resolution
        keyres = int(x.resolution.split("x")[0])
        keyftype = preftype == x.extension
        strict = (key3d, keyftype, keyres)
        nonstrict = (key3d, keyres, keyftype)
        return strict if ftypestrict else nonstrict

    r = min(streams, key=_sortkey)
    if ftypestrict and preftype != "any" and r.extension != preftype:
        return None
    return r


class VideoStream:
    """Represents a particular video stream at a given uri"""

    __UDT__ = VideoStreamType()

    def __init__(self, uri: str):
        self.uri = uri

    def __repr__(self) -> str:
        return f"VideoStream(uri={self.uri})"

    def _repr_html_(self):
        """TODO: codec"""
        from IPython.display import Video

        embed = True
        if urlparse(self.uri).scheme.lower().startswith("http"):
            embed = False
        return Video(self.uri, embed=embed, width=480, height=320)._repr_html_()

    def __eq__(self, other) -> bool:
        return isinstance(other, VideoStream) and self.uri == other.uri


class Segment:
    """A video segment bounded by frame numbers"""

    __UDT__ = SegmentType()

    def __init__(self, start_fno: int, end_fno: int):
        """

        Parameters
        ----------
        start_fno: int
            The starting frame number (0-indexed)
        end_fno: int
            The ending frame number. If <0 then it means end of the video

        Notes
        -----
        `fno` terminology is chosen to be consistent with the opencv library.
        """
        if start_fno < 0:
            raise ValueError("Cannot start with negative frame number")
        if end_fno > 0 and end_fno < start_fno:
            raise ValueError(
                "Ending frame must be negative or larger than " "starting frame"
            )
        self.start_fno = start_fno
        self.end_fno = end_fno

    def __repr__(self) -> str:
        return f"Segment(start_fno={self.start_fno}, end_fno={self.end_fno})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Segment) and (self.start_fno, self.end_fno) == (
            other.start_fno,
            other.end_fno,
        )


class VideoSampler(ABC):
    """
    Subclasses will implement different ways to retrieve samples from a given
    VideoStream.
    """

    def __init__(self, stream: VideoStream):
        self.stream = stream

    @abstractmethod
    def __iter__(self):
        pass


class SingleFrameSampler(VideoSampler):
    """
    A simple sampler that just returns one out of every `sample_rate` frames
    """

    def __init__(
        self, stream: VideoStream, sample_rate: int, start_frame: int, max_samples: int
    ):
        """
        Parameters
        ----------
        sample_rate: int
            The sampling rate in number of frames
        start_frame: int
            Start from a specific frame (0-based indexing)
        max_samples: int
            Yield at most this many frames (-1 means no max)
        """
        super().__init__(stream)
        self.sample_rate = sample_rate
        self.start_frame = start_frame
        self.max_samples = max_samples

    def __iter__(self):
        # TODO use seek for sparse sampling and maybe multithreaded
        cap = cv2.VideoCapture(self.stream.uri)
        if self.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        fno = 0
        success = cap.grab()  # get the next frame
        tot_samples = 0
        while success:
            if fno % self.sample_rate == 0:
                _, img = cap.retrieve()
                tot_samples += 1
                yield img
            if self.max_samples > -1 and tot_samples >= self.max_samples:
                break
            success = cap.grab()


class SamplerGenerator(ABC):
    @abstractmethod
    def get_sampler(self, stream: VideoStream) -> VideoSampler:
        pass


class SingleFrameGenerator(SamplerGenerator):
    """
    Generates samplers that just returns single frames
    """

    def __init__(
        self, sample_rate: int = 1, start_frame: int = 0, max_samples: int = -1
    ):
        """
        Parameters
        ----------
        sample_rate: int
            The sampling rate in number of frames
        start_frame: int
            Start from a specific frame (0-based indexing)
        max_samples: int
            Yield at most this many frames (-1 means no max)
        """
        self.sample_rate = sample_rate
        self.start_frame = start_frame
        self.max_samples = max_samples

    def get_sampler(self, stream):
        return SingleFrameSampler(
            stream, self.sample_rate, self.start_frame, self.max_samples
        )
