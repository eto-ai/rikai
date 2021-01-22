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

"""Vision Related User-defined Types:

- :py:class:`Image`
- :py:class:`Label`
"""

# Third-party libraries
import numpy as np
from PIL import Image as PILImage

# Rikai
from rikai.mixin import Asset, ToNumpy
from rikai.spark.types import ImageType, LabelType

__all__ = ["Image", "Label"]


class Image(ToNumpy, Asset):
    """An Image Asset.

    Parameters
    ----------
    uri : str
        The URI pointed to an Image stored on external storage.
    """

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
        """Convert this image into an :py:class:`numpy.ndarray`."""
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
