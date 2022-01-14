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

"""Geometry related PySpark UDFs"""

# Third Party
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# Rikai
from rikai.logging import logger
from rikai.spark.types.geometry import Box2dType
from rikai.types.geometry import Box2d

__all__ = ["area", "box2d", "box2d_from_center", "box2d_from_top_left"]


@udf(returnType=Box2dType())
def box2d(coords) -> Box2d:
    """Build a Box2d from ``[xmin,ymin,xmax,ymax]`` array."""
    return Box2d(coords[0], coords[1], coords[2], coords[3])


@udf(returnType=Box2dType())
def box2d_from_center(coords) -> Box2d:
    """Build a Box2d from a center-point based coordinate array:
    ``[center_x, center_y, width, height]``.

    See Also
    --------
    :py:meth:`rikai.types.geometry.Box2d.from_center`
    """
    return Box2d.from_center(coords[0], coords[1], coords[2], coords[3])


@udf(returnType=Box2dType())
def box2d_from_top_left(coords) -> Box2d:
    """Build :py:class:`Box2d` from a top-left based coordinate array:
    ``[x0, y0, width, height]``

    Example
    -------

    `Coco dataset <https://cocodataset.org/>`_ is one public dataset that use
    top-left coordinates


    >>> #! pyspark --packages ai.eto:rikai_2.12:0.0.1
    >>> import json
    >>> from rikai.spark.functions.geometry import box2d_from_top_left
    >>>
    >>> with open("coco_sample/annotations/train_sample.json") as fobj:
    ...     coco = json.load(fobj)
    >>> anno_df = (
    ...     spark
    ...     .createDataFrame(coco["annotations"][:5])
    ...     .withColumn("box2d", box2d_from_top_left("bbox"))
    ... )
    >>> anno_df.show()
    +--------------------+-----------+--------+--------------------+
    |                bbox|category_id|image_id|               box2d|
    +--------------------+-----------+--------+--------------------+
    |[505.24, 0.0, 47....|         72|  318219|Box2d(x=505.24, y...|
    |[470.68, 0.0, 45....|         72|  318219|Box2d(x=470.68, y...|
    |[442.51, 0.0, 43....|         72|  318219|Box2d(x=442.51, y...|
    |[380.74, 112.85, ...|         72|  554625|Box2d(x=380.74, y...|
    |[339.13, 32.99, 3...|         72|  554625|Box2d(x=339.13, y...|
    +--------------------+-----------+--------+--------------------+
    >>> anno_df.printSchema()
    root
    |-- bbox: array (nullable = true)
    |    |-- element: double (containsNull = true)
    |-- category_id: long (nullable = true)
    |-- image_id: long (nullable = true)
    |-- box2d: box2d (nullable = true)

    See Also
    --------
    :py:meth:`rikai.types.geometry.Box2d.from_top_left`.
    """
    return Box2d.from_top_left(coords[0], coords[1], coords[2], coords[3])


@udf(returnType=FloatType())
def area(bbox: Box2d) -> float:
    """A UDF to calculate the area of a bounding box."""
    return bbox.area
