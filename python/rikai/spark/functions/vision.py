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
from typing import Union

# Third Party
import cv2
import numpy as np
from pyspark.sql.functions import (
    udf,
    col,
    collect_list,
    lag,
    struct,
    row_number,
)
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    MapType,
    StructField,
    StringType,
    StructType,
)

# Rikai
from rikai.io import copy as _copy
from rikai.logging import logger
from rikai.numpy import ndarray
from rikai.spark.types.vision import ImageType
from rikai.types.vision import Image
from rikai.types.geometry import Box2d
from rikai.spark.types.geometry import Box2dType
from rikai.types.video import (
    YouTubeVideo,
    VideoStream,
    SingleFrameSampler,
    Segment,
)


__all__ = [
    "image",
    "image_copy",
    "motion_model",
    "numpy_to_image",
    "video_to_images",
    "spectrogram_image",
    "tracker_match",
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


@udf(returnType=MapType(IntegerType(), IntegerType()))
def tracker_match(trackers, detections, bbox_col="bbox", threshold=0.3):
    """
    Match Bounding Boxes across successive image frames.

    Parameters
        ----------
        trackers : List of Box2dType with str identifier
            A column of tracked objects.
        detections: List of Box2dType without tracker id matching
            The list of unmatched detections.
        bbox_col: str
                A string to name the column of bounding boxes.
        threshold : Float
                IOU of Box2d objects exceeding threshold will be matched.

        Return
        ------
        MapType
            Returns a MapType matching indices of trackers and detections.
    """
    from scipy.optimize import linear_sum_assignment

    similarity = lambda a, b: a.iou(b)
    if not trackers or not detections:
        return {}
    if len(trackers) == len(detections) == 1:
        if (
            similarity(trackers[0][bbox_col], detections[0][bbox_col])
            >= threshold
        ):
            return {0: 0}
        else:
            return {}

    sim_mat = np.array(
        [
            [
                similarity(tracked[bbox_col], detection[bbox_col])
                for tracked in trackers
            ]
            for detection in detections
        ],
        dtype=np.float32,
    )

    matched_idx = linear_sum_assignment(-sim_mat)
    matches = []
    for m in matched_idx:
        try:
            if sim_mat[m[0], m[1]] >= threshold:
                matches.append(m.reshape(1, 2))
        except:
            pass

    if len(matches) == 0:
        return {}
    else:
        matches = np.concatenate(matches, axis=0, dtype=int)

    rows, cols = zip(*np.where(matches))
    idx_map = {cols[idx]: rows[idx] for idx in range(len(rows))}
    return idx_map


@udf(returnType=ArrayType(Box2dType()))
def motion_model(frame, prev_frame, bboxes):
    if not prev_frame:
        prev_frame = frame
    gray = cv2.cvtColor(frame.to_numpy(), cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame.to_numpy(), cv2.COLOR_BGR2GRAY)
    inst = cv2.DISOpticalFlow.create(cv2.DISOpticalFlow)
    inst.setUseSpatialPropagation(False)
    flow = inst.calc(prev_gray, gray, None)
    h, w = flow.shape[:2]
    shifted_boxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        img_crop = flow[int(ymin) : int(ymax), int(xmin) : int(xmax), :]
        avg_y, avg_x = np.mean(flow[:, :, 0]), np.mean(flow[:, :, 1])
        shifted_boxes.append(
            Box2d(
                xmin=max(0, xmin + avg_x),
                ymin=max(0, ymin + avg_y),
                xmax=min(w, xmax + avg_x),
                ymax=min(h, ymax + avg_y),
            )
        )
    return shifted_boxes
