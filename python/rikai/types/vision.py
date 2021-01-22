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
import os
import pathlib

# 3rd party
import numpy as np
from PIL import Image as PILImage

# Rikai
from rikai.mixin import Asset, ToNumpy, Displayable
from rikai.spark.types import ImageType, LabelType
from rikai.types.geometry import Box2d

__all__ = ["Image", "Label", "Annotation"]


class Image(ToNumpy, Asset, Displayable):
    """Image resource"""

    __UDT__ = ImageType()

    def __init__(self, uri: str):
        super().__init__(uri)
        self._cached_data = None

    def show(self, **kwargs):
        """
        Custom visualizer for this image in jupyter notebook

        Parameters
        ----------
        kwargs: dict
            Optional display arguments

        Returns
        -------
        img: IPython.display.Image
        """
        from IPython.display import Image

        return Image(self.uri, **kwargs)

    def __repr__(self) -> str:
        return f"Image(uri={self.uri})"

    def _repr_html_(self):
        """Default visualizer for remote ref (or local ref under cwd)"""
        return self.show()._repr_html_()

    def _repr_mimebundle_(self, include=None, exclude=None):
        """default visualizer for embedded mime bundle"""
        return self.show()._repr_mimebundle_(include=include, exclude=exclude)

    def _repr_jpeg_(self):
        """default visualizer for embedded jpeg"""
        return self.show()._repr_jpeg_()

    def _repr_png_(self):
        """default visualizer for embedded png"""
        return self.show()._repr_png_()

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
