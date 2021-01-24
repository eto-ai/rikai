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

"""Vision-related Pyspark UDFs
"""

# Third Party
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, StringType

# Rikai
from rikai.logging import logger
from rikai.io import copy as _copy
from rikai.spark.types.vision import ImageType, LabelType
from rikai.types.geometry import Box2d
from rikai.types.vision import Image, Label


@udf(returnType=LabelType())
def label(value: str) -> Label:
    """Convert a string value into :py:class:`Label`."""
    return Label(value)


@udf(returnType=FloatType())
def area(bbox: Box2d) -> float:
    """A UDF to calculate the area of a bounding box"""
    return bbox.area


@udf(returnType=ImageType())
def image_copy(image: Image, uri: str) -> Image:
    """Copy the image to a new destination, specified by the URI.

    Parameters
    ----------
    image : Image
        An image object
    uri : str
        The base directory to copy the image to.

    Return
    ------
    Image
        Return a new image pointed to the new URI
    """
    logger.info("Copying image src=%s dest=%s", image.uri, uri)
    return Image(_copy(image.uri, uri))
