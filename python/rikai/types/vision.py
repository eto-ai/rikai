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

from io import BytesIO, IOBase
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, Sequence, Union
from urllib.parse import urlparse

# Third-party libraries
import numpy as np
from PIL import Image as PILImage

# Rikai
from rikai.conf import CONF_RIKAI_IMAGE_DEFAULT_FORMAT, options
from rikai.internal.uri_utils import normalize_uri
from rikai.io import copy
from rikai.mixin import Asset, Displayable, ToNumpy, ToPIL
from rikai.spark.types import ImageType
from rikai.types.geometry import Box2d

__all__ = ["Image"]


class Image(ToNumpy, ToPIL, Asset, Displayable):
    """An external Image Asset.

    It contains a reference URI to an image stored on the remote system.

    Parameters
    ----------
    image : bytes, file-like object, str or :py:class:`~pathlib.Path`
        It can be the content of image, or a URI / Path of an image.
    """

    __UDT__ = ImageType()

    def __init__(
        self,
        image: Union[bytes, bytearray, IOBase, str, Path],
    ):
        data, uri = None, None
        if isinstance(image, IOBase):
            data = image.read()
        elif isinstance(image, (bytes, bytearray)):
            data = image
        else:
            uri = image
        super().__init__(data=data, uri=uri)

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        uri: Optional[Union[str, Path]] = None,
        mode: Optional[str] = None,
        format: Optional[str] = None,
        **kwargs,
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
        format : str, optional
            The image format to save as. See
            `supported formats <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save>`_ for details.
        kwargs : dict, optional
            Optional arguments to pass to `PIL.Image.save <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save>`_.

        See Also
        --------
        :py:class:`PIL.Image.fromarray`
        :py:func:`~rikai.spark.functions.vision.numpy_to_image`

        """  # noqa: E501

        assert array is not None
        with PILImage.fromarray(array, mode=mode) as img:
            return cls.from_pil(img, uri, format=format, **kwargs)

    @staticmethod
    def from_pil(
        img: PILImage,
        uri: Optional[Union[str, Path]] = None,
        format: Optional[str] = None,
        **kwargs,
    ) -> Image:
        """Create an image in memory from a :py:class:`PIL.Image`.

        Parameters
        ----------
        img : :py:class:`PIL.Image`
            An PIL Image instance
        uri : str or Path
            The URI to store the image externally.
        format : str, optional
            The image format to save as. See
            `supported formats <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save>`_ for details.
        kwargs : dict, optional
            Optional arguments to pass to `PIL.Image.save <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save>`_.
        """  # noqa: E501

        format = format if format else options.rikai.image.default.format

        if uri is None:
            buf = BytesIO()
            img.save(buf, format=format, **kwargs)
            return Image(buf.getvalue())

        parsed = urlparse(normalize_uri(uri))
        if parsed.scheme == "file":
            img.save(uri, format=format, **kwargs)
        else:
            with NamedTemporaryFile() as fobj:
                img.save(fobj, format=format, **kwargs)
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
        with self.to_pil() as pil_img:
            return np.asarray(pil_img)

    def crop(
        self, box: Union[Box2d, List[Box2d]], format: Optional[str] = None
    ) -> Union[Image, List[Image]]:
        """Crop image specified by the bounding boxes, and returns the cropped
        images.

        Support crop images in batch, to save I/O overhead to download
        the original image.

        Parameters
        ----------
        box : :py:class:`Box2d` or :py:class:`List[Box2d]`
            The bounding box(es) to crop out of this image.
        format : str, optional
            The image format to save as

        Returns
        -------
        :py:class:`Image` or a list of :py:class:`Image`
        """
        if isinstance(box, Box2d):
            with self.to_pil() as pil_image:
                return Image.from_pil(pil_image.crop(box))

        assert isinstance(box, Sequence)
        crops = []
        with self.to_pil() as pil_image:
            for bbox in box:
                with pil_image.crop(bbox) as patch:
                    crops.append(Image.from_pil(patch))
        return crops
