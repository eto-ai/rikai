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
from rikai.spark.types import BBoxType, ImageType, LabelType

__all__ = ["Image", "BBox", "Label"]


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


class BBox(ToNumpy):
    """2D Bounding Box."""

    __UDT__ = BBoxType()

    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float):
        assert 0 <= xmin <= xmax, f"xmin({xmin}) is not smaller than xmax({xmax})"
        assert 0 <= ymin <= ymax, f"xmin({ymin}) is not smaller than xmax({ymax})"

        self.xmin = float(xmin)
        self.ymin = float(ymin)
        self.xmax = float(xmax)
        self.ymax = float(ymax)

    def __repr__(self) -> str:
        return f"BBox({self.xmin}, {self.ymin}, {self.xmax}, {self.ymax})"

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, BBox)
            and o.xmin == self.xmin
            and o.ymin == self.ymin
            and o.xmax == self.xmax
            and o.ymax == self.ymax
        )

    def to_numpy(self) -> np.ndarray:
        """Convert BBox to numpy ndarray"""
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax])

    @property
    def area(self) -> float:
        """Area of the bounding box"""
        return abs(self.xmax - self.xmin) * abs(self.ymax - self.ymin)

    def iou(self, other: "BBox") -> float:
        """Compute intersection over union(IOU)."""
        assert isinstance(
            other, BBox
        ), f"Can only compute iou between BBox, got {type(other)}"
        # Find intersection
        xmin = max(self.xmin, other.xmin)
        ymin = max(self.ymin, other.ymin)
        xmax = min(self.xmax, other.xmax)
        ymax = min(self.ymax, other.ymax)
        inter_area = max(0, xmax - xmin) * max(0, ymax - ymin)

        try:
            return inter_area / (self.area + other.area - inter_area)
        except ZeroDivisionError:
            return 0


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

    def __init__(self, label: Label, text: str, bbox: BBox, score: float = 0.0):
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
