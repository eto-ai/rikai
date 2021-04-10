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

try:
    import ffmpeg
    use_ffmpeg = True
except ImportError:
    import cv2
    use_ffmpeg = False

from rikai.mixin import Displayable
from rikai.spark.types import SegmentType, VideoStreamType, YouTubeVideoType

from rikai.video.ffmpeg import VideoFrameSampler

__all__ = [
    "YouTubeVideo",
    "VideoStream",
    "VideoSampler",
    "SingleFrameSampler",
    "Segment",
    "VideoFrameSampler",
]


class YouTubeVideo(Displayable):
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

    def display(self, width: int = 400, height: int = 300, **kwargs):
        """
        Visualization in jupyter notebook with custom options

        Parameters
        ----------
        width: int, default 400
            Width in pixels
        height: int, default 300
            Height in pixels
        kwargs: dict
            See :py:class:`IPython.display.YouTubeVideo` for other kwargs

        Returns
        -------
        v: IPython.display.YouTubeVideo
        """
        from IPython.display import YouTubeVideo

        return YouTubeVideo(self.vid, width=width, height=height, **kwargs)

    def _repr_html_(self):
        """default visualization in jupyter notebook cell"""
        return self.display()._repr_html_()

    def __eq__(self, other) -> bool:
        return isinstance(other, YouTubeVideo) and self.vid == other.vid

    def get_stream(
        self, ext: str = "mp4", quality: str = "worst"
    ) -> "VideoStream":
        """
        Get a reference to a particular stream

        Parameters
        ----------
        ext: str, default 'mp4'
            The preferred extension type to get. One of ['ogg', 'm4a', 'mp4',
            'flv', 'webm', '3gp']
            See: https://pythonhosted.org/Pafy/#Pafy.Stream.extension
        quality: str, default 'worst'
            Either 'worst' (lowest bitrate) or 'best' (highest bitrate)
            See: https://pythonhosted.org/Pafy/index.html#Pafy.Pafy.getbest

        Returns
        -------
        v: VideoStream
            VideoStream referencing an actual video resource
        """
        try:
            import pafy
        except ImportError as e:
            print(
                "Run `pip install rikai[youtube] to install pafy and "
                "youtube_dl to work with youtube videos."
            )
            raise e
        ext, quality = ext.strip().lower(), quality.strip().lower()
        if quality == "worst":
            stream = getworst(pafy.new(self.uri), preftype=ext)
        else:
            stream = pafy.new(self.uri).getbest(preftype=ext)
        return VideoStream(stream.url)


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


class VideoStream(Displayable):
    """Represents a particular video stream at a given uri"""

    __UDT__ = VideoStreamType()

    def __init__(self, uri: str):
        self.uri = uri

    def __repr__(self) -> str:
        return f"VideoStream(uri={self.uri})"

    def display(self, width: int = None, height: int = None, **kwargs):
        """
        Customize visualization in jupyter notebook

        Parameters
        ----------
        width: int, default None
            Width in pixels. Defaults to the original video width
        height: int, default None
            Height in pixels. Defaults to the original video height
        kwargs: dict
            See :py:class:`IPython.display.Video` doc for other kwargs

        Returns
        -------
        v: IPython.display.Video
        """
        from IPython.display import Video

        return Video(self.uri, width=width, height=height, **kwargs)

    def _repr_html_(self):
        """default visualizer for jupyter notebook"""
        return self.display()._repr_html_()

    def __eq__(self, other) -> bool:
        return isinstance(other, VideoStream) and self.uri == other.uri

    def __iter__(self):
        """Iterate through every frame in the video"""

        sampler = VideoFrameSampler(self) if use_ffmpeg else SingleFrameSampler(self)
        for frame in sampler:
            yield frame


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
                "Ending frame must be negative or larger than starting frame"
            )
        self.start_fno = start_fno
        self.end_fno = end_fno

    def __repr__(self) -> str:
        return f"Segment(start_fno={self.start_fno}, end_fno={self.end_fno})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Segment) and (
            self.start_fno,
            self.end_fno,
        ) == (
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
        self,
        stream: VideoStream,
        sample_rate: int = 1,
        start_frame: int = 0,
        max_samples: int = -1,
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

class VideoFrameSampler(VideoSampler):
    """
    An Image sampler returning one from every `sample_rate` video frames.
    """

    def __init__(
        self,
        stream: VideoStream,
        sample_rate: int = 1,
        start_frame: int = 0,
        max_samples: int = -1,
        left: int = 0,
        top: int = 0,
        crop_width: int = -1,
        crop_height: int = -1,
        scale_width: int = -1,
        scale_height: int = -1,
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
        self.left = (left,)
        self.top = (top,)
        self.crop_width = (crop_width,)
        self.crop_height = (crop_height,)
        self.scale_width = (scale_width,)
        self.scale_height = (scale_height,)
        self.ffmpeg_load = None
        self.probe = None

    def _probe(self):
        try:
            self.probe = ffmpeg.probe(self.stream.uri)
        except ffmpeg.Error as e:
            print(e.stderr, file=sys.stderr)
            sys.exit(1)
        video_stream = next(
            (
                stream
                for stream in self.probe["streams"]
                if stream["codec_type"] == "video"
            ),
            None,
        )
        if video_stream is None:
            print("No video stream found", file=sys.stderr)
            sys.exit(1)

        self.width = int(video_stream["width"])
        self.height = int(video_stream["height"])
        self.num_frames = int(video_stream["nb_frames"])
        self.duration = int(video_stream["duration"])
        self.frame_rate = np.round(self.duration / self.num_frames).astype(int)

    def _ffmpeg_load(self):
        if not self.probe:
            self._probe()
        self.ffmpeg_load = (
            ffmpeg.input(self.stream.uri)
            .crop(self.left, self.top, crop_width, crop_width)
            .filter("scale", self.scale_width, self.scale_height)
        )

    def sample_frame(self, frame_no=None, vcodec="mjpeg"):
        if not self.ffmpeg_load:
            self._ffmpeg_load()
        if frame_no is None:
            lower_limit = upper_limit = self.start_frame
            upper_limit += (
                self.num_frames if not self.max_samples else self.max_samples
            )
        frame_no = np.random.randint(lower_limit, upper_limit)

        img_bytes, _ = (
            self.ffmpeg_load.filter("select", "gte(n,{})".format(frame_no))
            .output("pipe:", vframes=1, format="image2", vcodec=vcodec)
            .run(capture_stdout=True)
        )

        return Image(data=img_bytes)

    def load_video(self):
        if not self.ffmpeg_load:
            self._ffmpeg_load()

        crop_width = self.crop_width if self.crop_width else self.width
        crop_width = self.crop_height if self.crop_height else self.height
        video_data, _ = self.ffmpeg_load.output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            start_number=self.start_frame,
            vframes=self.start_frame + self.max_samples
            if self.max_samples
            else None,
            r=1 / self.sample_rate,
        ).run(capture_stdout=True)
        h = self.crop_height if not self.scale_height else self.scale_height
        w = self.crop_width if not self.scale_width else self.scale_width
        img_size = h * w * 3
        return [Image(data=video_data[i:i+img_size]) for i in range(0, len(video_data), img_size)]

    def __iter__(self):
        for frame in iter(self.load_video()):
            yield frame

