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

# Third Party
from pyspark.sql.functions import udf

# Rikai
from rikai.io import copy as _copy
from rikai.logging import logger
from rikai.numpy import ndarray
from rikai.spark.types.vision import ImageType
from rikai.types.vision import Image

__all__ = ["image", "image_copy", "numpy_to_image"]


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
