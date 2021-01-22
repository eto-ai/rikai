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

"""Geometry types
"""

import numpy as np

from rikai.mixin import ToNumpy
from rikai.spark.types.geometry import PointType, Box3dType, Box2dType

__all__ = ["Point", "Box3d", "Box2d"]


class Point(ToNumpy):
    """Point in a 3-D space, specified by ``(x, y, z)`` coordinates.

    Attributes
    ----------
    x : float
        The X coordinate.
    y : float
        The Y coordinate.
    z : float
        The Z coordinate.
    """

    __UDT__ = PointType()

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y}, {self.z})"

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, Point) and self.x == o.x and self.y == o.y and self.z == o.z
        )

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


class Box2d(ToNumpy):
    """2-D Bounding Box.

    Attributes
    ----------
    x : float
        X-coordinate of the center point of the box
    y : float
        Y-coordinate of the center point of the box
    width : float
        The width of the box
    height : float
        The height of the box
    """

    __UDT__ = Box2dType()

    def __init__(self, x: float, y: float, width: float, height: float):

        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.height = float(height)

    def __repr__(self) -> str:
        return f"Box2d(x={self.x}, y={self.y}, w={self.width}, h={self.height})"

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, Box2d)
            and o.x == self.x
            and o.y == self.y
            and o.width == self.width
            and o.height == self.height
        )

    def to_numpy(self) -> np.ndarray:
        """Convert a :py:class:`Box2d` to numpy ndarray"""
        return np.array([self.x, self.y, self.width, self.height])

    @property
    def xmin(self) -> float:
        return self.x - self.width / 2

    @property
    def xmax(self) -> float:
        return self.x + self.width / 2

    @property
    def ymin(self) -> float:
        return self.y - self.height / 2

    @property
    def ymax(self) -> float:
        return self.y + self.height * 2

    @property
    def area(self) -> float:
        """Area of the bounding box"""
        return self.width * self.height

    def iou(self, other: "Box2d") -> float:
        """Compute intersection over union(IOU)."""
        assert isinstance(
            other, Box2d
        ), f"Can only compute iou between Box2d, got {type(other)}"
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


class Box3d(ToNumpy):
    """A 3-D bounding box

    Attributes
    ----------
    center : Point
        Center :py:class:`Point` of the bounding box
    length : float
        The x dimention of the box
    width : float
        The y dimention of the box
    height : float
        The z dimention of the box
    heading : float
        The heading of the bounding box (in radians).  The heading is the angle
        required to rotate +x to the surface normal of the box front face. It is
        normalized to ``[-pi, pi)``.

    References
    ----------
    * Waymo Dataset Spec https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto
    """

    __UDT__ = Box3dType()

    def __init__(
        self,
        center: Point,
        length: float,
        width: float,
        height: float,
        heading: float,
    ):
        self.center = center
        self.length = float(length)
        self.width = float(width)
        self.height = float(height)
        self.heading = float(heading)

    def __repr__(self) -> str:
        return f"Box3d(center={self.center}, l={self.length}, h={self.height}, w={self.weight}, heading={self.heading})"

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, Box3d)
            and self.center == o.center
            and self.length == o.length
            and self.width == o.width
            and self.height == o.height
            and self.heading == o.heading
        )

    def to_numpy(self) -> np.ndarray:
        return None
