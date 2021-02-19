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
"""

from __future__ import annotations

from pathlib import Path
from typing import Union
from urllib.parse import urlparse
from tempfile import NamedTemporaryFile

# Third-party libraries
import numpy as np
from PIL import Image as PILImage

# Rikai
from rikai.internal.uri_utils import normalize_uri
from rikai.mixin import Asset, Displayable, ToNumpy
from rikai.spark.types import ImageType
from rikai.io import copy

__all__ = ["Image"]


class Image(ToNumpy, Asset, Displayable):
    """An external Image Asset.

    It contains a reference URI to an image stored on the remote system.

    Parameters
    ----------
    uri : str
        The URI pointed to an Image stored on external storage.
    """

    __UDT__ = ImageType()

    def __init__(self, uri: Union[str, Path]):
        super().__init__(uri)
        self._cached_data = None

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        uri: Union[str, Path],
        mode: str = None,
    ) -> Image:
        """Create an image in memory from numpy array.

        Parameters
        ----------
        array : np.ndarray
            Array data
        uri : str or Path
            The external URI to store the data.
        mode : str, optional
            The mode which PIL used to create image. See supported
            `modes on PIL document <https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes>`_.

        See Also
        --------
        :py:class:`PIL.Image.fromarray`
        :py:func:`~rikai.spark.functions.vision.numpy_to_image`

        """  # noqa: E501

        assert array is not None
        img = PILImage.fromarray(array, mode=mode)
        return cls.from_pil(img, uri)

    @staticmethod
    def from_pil(img: PILImage, uri: Union[str, Path]) -> Image:
        """Create an image in memory from a :py:class:`PIL.Image`.

        Parameters
        ----------
        img : :py:class:`PIL.Image`
            An PIL Image instance
        uri : str or Path
            The URI to store the image externally.
        """
        parsed = urlparse(normalize_uri(uri))
        if parsed.scheme == "file":
            img.save(uri)
        else:
            with NamedTemporaryFile() as fobj:
                img.save(fobj)
                fobj.flush()
                copy(fobj.name, uri)
        return Image(uri)

    def display(self, **kwargs):
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

        with self.open() as fobj:
            return Image(fobj.read(), **kwargs)

    def __repr__(self) -> str:
        return f"Image(uri={self.uri})"

    def _repr_html_(self):
        """Default visualizer for remote ref (or local ref under cwd)"""
        return self.display()._repr_html_()

    def _repr_mimebundle_(self, include=None, exclude=None):
        """default visualizer for embedded mime bundle"""
        return self.display()._repr_mimebundle_(
            include=include, exclude=exclude
        )

    def _repr_jpeg_(self):
        """default visualizer for embedded jpeg"""
        return self.display()._repr_jpeg_()

    def _repr_png_(self):
        """default visualizer for embedded png"""
        return self.display()._repr_png_()

    def __eq__(self, other) -> bool:
        return isinstance(other, Image) and super().__eq__(other)

    def to_pil(self) -> PILImage:
        """Return an PIL image.

        Note
        ----
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
