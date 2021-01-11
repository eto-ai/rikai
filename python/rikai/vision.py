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

"""Vision Related Types
"""
from PIL import Image as PILImage

import numpy as np
from rikai.mixin import Asset, ToNumpy
from rikai.spark.types import ImageType, LabelType
from rikai.types.geometry import Box2d

from rikai.spark.types import (
    ImageType,
    LabelType,
    YouTubeVideoType,
    VideoStreamType,
)

__all__ = ["Image", "Box2d", "Label", "YouTubeVideo", "VideoSegment", "VideoStream"]


class Image(ToNumpy, Asset):
    """Image resource"""

    __UDT__ = ImageType()

    def __init__(self, uri: str):
        super().__init__(uri)
        self._cached_data = None

    def __repr__(self) -> str:
        return f"Image(uri={self.uri})"

    def _repr_html_(self):
        """Jupyter integration

        TODO: find more appropriate way to return image in jupyter?
        """
        return f"<img src={self.uri} />"

    def __eq__(self, other) -> bool:
        return isinstance(other, Image) and super().__eq__(other)

    def to_pil(self) -> PILImage:
        """Return an PIL image.

        The caller should close the image.
        https://pillow.readthedocs.io/en/stable/reference/open_files.html#image-lifecycle
        """
        return PILImage.open(self.open())

    def to_numpy(self) -> np.ndarray:
        """Convert image into an numpy array."""
        if self._cached_data is None:
            with self.to_pil() as pil_img:
                self._cached_data = np.asarray(pil_img)
        assert self._cached_data is not None
        return self._cached_data


class Label(ToNumpy):
    """Text Label

    A strong-typed Text label.
    """

    __UDT__ = LabelType()

    def __init__(self, label: str):
        """Label

        Parameters
        ----------
        label : str
            Label text

        """
        self.label = label

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        return f"label({self.label})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Label) and self.label == other.label

    def to_numpy(self) -> np.ndarray:
        return np.array([self.label])


class Annotation(ToNumpy):
    """Vision detection annotation"""

    __UDT__ = LabelType()

    def __init__(self, label: Label, text: str, bbox: Box2d, score: float = 0.0):
        self.label = label
        self.text = text
        self.bbox = bbox
        self.score = score

    def to_numpy(self) -> np.ndarray:
        return {
            "label": self.label.to_numpy(),
            "text": self.text,
            "bbox": self.bbox.to_numpy(),
            "score": self.score,
        }


class VideoStream:
    """VideoStream resource"""

    __UDT__ = VideoStreamType()

    def __init__(self, uri: str):
        self.uri = uri

    def __repr__(self) -> str:
        return f"VideoStream(uri={self.uri})"

    def _repr_html_(self):
        """TODO: codec"""
        from IPython.display import Video

        if pathlib.Path(self.uri).exists():
            path = pathlib.Path(self.uri).relative_to(os.getcwd())
            return Video(path, width=480, height=320)._repr_html_()
        else:
            return Video(self.uri, width=480, heigh=320)._repr_html_()

    def __eq__(self, other) -> bool:
        return isinstance(other, VideoStream) and self.uri == other.uri


class YouTubeVideo:

    __UDT__ = YouTubeVideoType()

    def __init__(self, vid: str):
        self.vid = vid
        self.uri = "https://www.youtube.com/watch?v={0}".format(self.vid)
        self.embed_url = "http://www.youtube.com/embed/{0}".format(self.vid)

    def __repr__(self) -> str:
        return "YouTubeVideo({0})".format(self.vid)

    def _repr_html_(self):
        from IPython.lib.display import YouTubeVideo

        return YouTubeVideo(self.embed_url)._repr_html_()

    def __eq__(self, other) -> bool:
        return isinstance(other, YouTubeVideo) and self.vid == other.vid
