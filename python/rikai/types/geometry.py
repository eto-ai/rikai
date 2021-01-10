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
from rikai.spark.types.geometry import PointType


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


class Box3d(ToNumpy):
    """A 3-D bounding box

    Attributes
    ----------
    center : Point
        Center point of the bounding box
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
    .. [1] Waymo Dataset Spec https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto
    """

    def __init__(
        self,
        center: Point,
        length: float,
        width: float,
        height: float,
        heading: float,
    ):
        self.center = center
        self.length = length
        self.width = width
        self.height = height
        self.heading = heading

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
