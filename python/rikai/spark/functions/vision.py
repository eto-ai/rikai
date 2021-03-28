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

"""Vision related Spark UDFs.
"""

# Standard library
import os
import subprocess
from typing import Union

# Third Party
import numpy as np
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType

# Rikai
from rikai.io import copy as _copy
from rikai.logging import logger
from rikai.numpy import ndarray
from rikai.spark.types.vision import ImageType, SegmentType
from rikai.types.vision import Image
from rikai.types.video import (
    YouTubeVideo,
    VideoStream,
    SingleFrameSampler,
    Segment,
)


__all__ = [
    "image",
    "image_copy",
    "numpy_to_image",
    "video_to_images",
    "spectrogram_image",
]


@udf(returnType=ImageType())
def image(uri: str) -> Image:
    """Build an :py:class:`Image` from a URI."""
    return Image(uri)


@udf(returnType=ImageType())
def image_copy(img: Image, uri: str) -> Image:
    """Copy the image to a new destination, specified by the URI.

    Parameters
    ----------
    img : Image
        An image object
    uri : str
        The base directory to copy the image to.

    Return
    ------
    Image
        Return a new image pointed to the new URI
    """
    logger.info("Copying image src=%s dest=%s", img.uri, uri)
    return Image(_copy(img.uri, uri))


@udf(returnType=ImageType())
def numpy_to_image(array: ndarray, uri: str) -> Image:
    """Convert a numpy array to image, and upload to external storage.

    Parameters
    ----------
    array : :py:class:`numpy.ndarray`
        Image data.
    uri : str
        The base directory to copy the image to.

    Return
    ------
    Image
        Return a new image pointed to the new URI.

    Example
    -------

    >>> spark.createDataFrame(..).registerTempTable("df")
    >>>
    >>> spark.sql(\"\"\"SELECT numpy_to_image(
    ...        resize(grayscale(image)),
    ...        lit('s3://asset')
    ...    ) AS new_image FROM df\"\"\")

    See Also
    --------
    :py:meth:`rikai.types.vision.Image.from_array`
    """
    return Image.from_array(array, uri)


@udf(returnType=ArrayType(ImageType()))
def video_to_images(
    video: Union[VideoStream, YouTubeVideo],
    output_uri: str,
    segment: Segment = Segment(0, -1),
    sample_rate: int = 1,
    max_samples: int = 15000,
    quality: str = "worst",
) -> list:
    """Extract video frames into a list of images.

    Parameters
    ----------
    video : Video
        An video object, either YouTubeVideo or VideoStream.
    output_uri: str
        Frames will be written as <output_uri>/<fno>.jpg
    segment: Segment, default Segment(0, -1)
        A Segment object, localizing video in time to (start_fno, end_fno)
    sample_rate : int, default 1
        Keep 1 out of every sample_rate frames.
    max_samples : int, default 15000
        Return at most this many frames (-1 means no max)
    quality: str, default 'worst'
        Either 'worst' (lowest bitrate) or 'best' (highest bitrate)
        See: https://pythonhosted.org/Pafy/index.html#Pafy.Pafy.getbest

    Return
    ------
    List
        Return a list of images from video indexed by frame number.
    """
    assert isinstance(
        video, (YouTubeVideo, VideoStream)
    ), "Input type must be YouTubeVideo or VideoStream"
    assert isinstance(segment, Segment), "Second input type must be Segment"

    start_frame = segment.start_fno
    if segment.end_fno > 0:
        max_samples = min((segment.end_fno - start_frame), max_samples)

    if isinstance(video, YouTubeVideo):
        video_iterator = SingleFrameSampler(
            video.get_stream(quality=quality),
            sample_rate,
            start_frame,
            max_samples,
        )
    else:
        video_iterator = SingleFrameSampler(
            video, sample_rate, start_frame, max_samples
        )

    return [
        Image.from_array(
            img,
            os.path.join(
                output_uri, "{}.jpg".format((start_frame + idx) * sample_rate)
            ),
        )
        for idx, img in enumerate(video_iterator)
    ]


@udf(returnType=ImageType())
def spectrogram_image(
    video: Union[VideoStream, YouTubeVideo],
    output_uri: str,
    segment: Segment = Segment(0, -1),
    size: int = 224,
    max_samples: int = 15000,
) -> Image:
    """Applies ffmpeg filter to generate spectrogram image.

    Parameters
    ----------
    video : VideoStream or YouTubeVideo
        A video object whose audio track will be converted to spectrogram
    output_uri: str
        The uri to which the spectrogram image will be written to
    segment: Segment
            A Segment object, localizing video in time to (start_fno, end_fno)
    max_samples : Int
            Yield at most this many frames (-1 means no max)
    size : Int
        Sets resolution of frequency, time spectrogram image.

    Return
    ------
    Image
        Return an Image of the audio spectrogram.
    """
    try:
        import ffmpeg
    except ImportError:
        raise ValueError(
            "Couldn't import ffmpeg. Please make sure to "
            "`pip install ffmpeg-python` explicitly or install "
            "the correct extras like `pip install rikai[all]`"
        )
    assert isinstance(
        video, (YouTubeVideo, VideoStream)
    ), "Input type must be YouTubeVideo or VideoStream"
    assert isinstance(segment, Segment), "Second input type must be Segment"

    start_frame = segment.start_fno
    if segment.end_fno > 0:
        max_samples = min((segment.end_fno - start_frame), max_samples)
    video_uri = (
        video.get_stream().uri
        if isinstance(video, YouTubeVideo)
        else video.uri
    )
    output, _ = (
        ffmpeg.input(video_uri)
        .filter("showspectrumpic", "{}x{}".format(size, size), legend=0)
        .output(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            start_number=start_frame,
            vframes=max_samples,
        )
        .run(capture_stdout=True)
    )
    return Image.from_array(
        np.frombuffer(output, np.uint8).reshape([size, size, 3]), output_uri
    )


@udf(returnType=ArrayType(SegmentType()))
def scene_segmentor(
    video: Union[VideoStream, YouTubeVideo],
    threshold: float = 0.3,
    fps: int = 30,
) -> list:
    """Extracts video segments from VideoStream or YouTubeVideo by thresholding sum-of-absolute differences.
    https://github.com/FFmpeg/FFmpeg/blob/master/libavfilter/f_select.c
    Parameters
    ----------
    video : Video
      An video object, either YouTubeVideo or VideoStream.
    threshold: float, default 0.3
      Scene detection score, values in [0, 1]
    fps: int, default 30
      Frames per second of video asset
    Return
    ------
    List
      Return a list of Segments from video.
    """
    try:
        import ffmpeg
    except ImportError:
        raise ValueError(
            "Couldn't import ffmpeg. Please make sure to "
            "`pip install ffmpeg-python` explicitly or install "
            "the correct extras like `pip install rikai[all]`"
        )
    assert isinstance(
        video, (YouTubeVideo, VideoStream)
    ), "Input type must be YouTubeVideo or VideoStream"

    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    video_uri = (
        video.get_stream().uri
        if isinstance(video, YouTubeVideo)
        else video.uri
    )

    p = subprocess.Popen(
        (
            ffmpeg.input(video_uri)
            .filter("select", "gte(scene,{})".format(threshold))
            .filter("showinfo")
            .output("-", format="null")
            .compile()
        ),
        stderr=subprocess.PIPE,
    )
    result = p.communicate()[1].decode("utf-8")
    cut_times = [0] + [
        float(ln.split()[0])
        for ln in result.split("pts_time:")
        if isfloat(ln.split()[0])
    ]
    if len(cut_times) == 1:
        return [Segment(0, -1)]
    intervals = [
        (cut_times[idx - 1], cut_times[idx])
        for idx, time in enumerate(cut_times)
    ][1:]
    return [
        Segment(start_fno=int(fps * tup[0]), end_fno=int(fps * tup[1]))
        for tup in intervals
        if tup[0] < tup[1]
    ]
