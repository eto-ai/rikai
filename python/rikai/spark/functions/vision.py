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
from pathlib import Path
from typing import Optional, Union

# Third Party
import numpy as np
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
from rikai.io import copy as _copy
from rikai.logging import logger
from rikai.numpy import ndarray
from rikai.spark.types.vision import ImageType
from rikai.types.video import (
    Segment,
    SingleFrameSampler,
    VideoStream,
    YouTubeVideo,
)
from rikai.types.vision import Image

__all__ = [
    "to_image",
    "image_copy",
    "numpy_to_image",
    "video_to_images",
    "spectrogram_image",
    "video_metadata",
    "video_metadata_schema",
]


@udf(returnType=ImageType())
def to_image(image_data: Union[bytes, bytearray, str, Path]) -> Image:
    """Build an :py:class:`Image` from
    bytes, file-like object, str, or :py:class:`~pathlib.Path`.

    Parameters
    ----------
    image_data : bytes, bytearray, str, Path
        The resource identifier or bytes of the source image.

    Return
    ------
    img: Image
        An Image from the given embedded data or URI

    Example
    -------

    >>> df = spark.read.format("image").load("<path-to-data>")
    >>>
    >>> df.withColumn("new_image", to_image("image.data"))
    """
    return Image(image_data)


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
def numpy_to_image(
    array: ndarray, uri: str, format: str = None, **kwargs
) -> Image:
    """Convert a numpy array to image, and upload to external storage.

    Parameters
    ----------
    array : :py:class:`numpy.ndarray`
        Image data.
    uri : str
        The base directory to copy the image to.
    format : str, optional
        The image format to save as. See
        `supported formats <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save>`_ for details.
    kwargs : dict, optional
        Optional arguments to pass to `PIL.Image.save <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save>`_.

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
    """  # noqa: E501
    return Image.from_array(array, uri, format=format, **kwargs)


@udf(returnType=ArrayType(ImageType()))
def video_to_images(
    video: Union[VideoStream, YouTubeVideo],
    output_uri: str,
    segment: Segment = Segment(0, -1),
    sample_rate: int = 1,
    max_samples: int = 15000,
    quality: str = "worst",
    image_format: str = "png",
    **image_kwargs,
) -> list:
    """Extract video frames into a list of images.

    Parameters
    ----------
    video : Video
        An video object, either YouTubeVideo or VideoStream.
    output_uri: str
        Frames will be written as <output_uri>/<fno>.<img_format>
    segment: Segment, default Segment(0, -1)
        A Segment object, localizing video in time to (start_fno, end_fno)
    sample_rate : int, default 1
        Keep 1 out of every sample_rate frames.
    max_samples : int, default 15000
        Return at most this many frames (-1 means no max)
    quality: str, default 'worst'
        Either 'worst' (lowest bitrate) or 'best' (highest bitrate)
        See: https://pythonhosted.org/Pafy/index.html#Pafy.Pafy.getbest
    image_format : str, optional
        The image format to save as. See
        `supported formats <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save>`_ for details.
    image_kwargs : dict, optional
        Optional arguments to pass to `PIL.Image.save <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save>`_.
    ------
    List
        Return a list of images from video indexed by frame number.
    """  # noqa: E501
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
                output_uri,
                "{}.{}".format(
                    (start_frame + idx) * sample_rate, image_format
                ),
            ),
            format=image_format,
            **image_kwargs,
        )
        for idx, img in enumerate(video_iterator)
    ]


video_metadata_schema = StructType(
    [
        StructField("width", IntegerType(), True),
        StructField("height", IntegerType(), True),
        StructField("num_frames", IntegerType(), True),
        StructField("duration", FloatType(), True),
        StructField("bit_rate", IntegerType(), True),
        StructField("frame_rate", IntegerType(), True),
        StructField("codec", StringType(), True),
        StructField("size", IntegerType(), True),
        StructField(
            "_errors",
            StructType(
                [
                    StructField("message", StringType(), True),
                    StructField("stderr", StringType(), True),
                    StructField("probe", StringType(), True),
                ]
            ),
            True,
        ),
    ]
)


@udf(returnType=video_metadata_schema)
def video_metadata(video: Union[str, VideoStream]) -> dict:
    """Return useful video stream metadata about the given video

    Parameters
    ----------
    video: str or VideoStream
        The video uri or the Video object

    Returns
    -------
    result: dict
        { width,
          height,
          num_frames,
          duration,
          bit_rate,
          frame_rate,
          codec,
          size,
          _errors }

    Notes
    -----
    The frame_rate is rounded to the nearest whole number of frames per sec

    Examples
    --------
    The following returns the fps rounded to the nearest integer
    ```
    import rikai.spark.functions as RF
    (spark.createDataFrame([(VideoStream(<uri>),)], ['video'])
          .withColumn('meta', RF.video_metadata('video'))
          .select('meta.data.frame_rate'))
    ```
    """
    probe_result = _probe(video)
    if probe_result.get("_errors", None):
        return probe_result

    video_stream = next(
        (
            stream
            for stream in probe_result.get("streams", [])
            if stream.get("codec_type", None) == "video"
        ),
        None,
    )
    if video_stream is None:
        return _error("No video stream found", probe=probe_result)

    try:
        return {
            "width": _int_or_none(video_stream, "width"),
            "height": _int_or_none(video_stream, "height"),
            "num_frames": _int_or_none(video_stream, "nb_frames"),
            "frame_rate": _fps_or_none(video_stream),
            "duration": _float_or_none(video_stream, "duration"),
            "bit_rate": _int_or_none(video_stream, "bit_rate"),
            "codec": video_stream.get("codec_name", None),
            "size": _int_or_none(probe_result.get("format", {}), "size"),
        }
    except Exception as e:
        return _error(str(e))


def _probe(video):
    try:
        import ffmpeg
    except ImportError as e:
        raise ImportError(
            "Couldn't import ffmpeg. Please make sure to "
            "`pip install ffmpeg-python` explicitly or install "
            "the correct extras like `pip install rikai[video]`"
        ) from e

    uri = video.uri if isinstance(video, VideoStream) else video
    try:
        return ffmpeg.probe(uri)
    except Exception as e:
        return _error(str(e), stderr=getattr(e, "stderr", None))


def _int_or_none(data, key):
    return int(data[key]) if key in data else None


def _float_or_none(data, key):
    return float(data[key] if key in data else None)


def _fps_or_none(data: dict) -> Optional[int]:
    # ffprobe returns frame rate in timebase units (as a fraction)
    avg_frame_rate = data.get("avg_frame_rate", None)
    if avg_frame_rate is None:
        return None
    n, d = [int(x) for x in avg_frame_rate.split("/")]
    if d == 0:
        raise ValueError("avg_frame_rate had 0 time base")
    return int(round(n / d))


def _error(message, stderr=None, probe=None):
    err = {"message": message}
    if stderr is not None:
        err["stderr"] = stderr.decode("utf-8", "backslashreplace")
    if probe is not None:
        err["probe"] = str(probe)
    return {"_errors": err}


@udf(returnType=ImageType())
def spectrogram_image(
    video: Union[VideoStream, YouTubeVideo],
    output_uri: str,
    segment: Segment = Segment(0, -1),
    size: int = 224,
    max_samples: int = 15000,
    image_format: str = None,
    **image_kwargs,
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
    image_format : str, optional
        The image format to save as. See
        `supported formats <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save>`_ for details.
    image_kwargs : dict, optional
        Optional arguments to pass to `PIL.Image.save <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save>`_.

    Return
    ------
    Image
        Return an Image of the audio spectrogram.
    """  # noqa: E501
    try:
        import ffmpeg
    except ImportError:
        raise ValueError(
            "Couldn't import ffmpeg. Please make sure to "
            "`pip install ffmpeg-python` explicitly or install "
            "the correct extras like `pip install rikai[video]`"
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
        np.frombuffer(output, np.uint8).reshape([size, size, 3]),
        output_uri,
        format=image_format,
        **image_kwargs,
    )
