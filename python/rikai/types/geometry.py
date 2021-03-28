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

from __future__ import annotations

from numbers import Real
from typing import List, Sequence, Tuple, Union

import numpy as np

from rikai.mixin import ToNumpy
from rikai.spark.types.geometry import Box2dType, Box3dType, PointType

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
        # pylint: disable=invalid-name
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y}, {self.z})"

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, Point)
            and self.x == o.x
            and self.y == o.y
            and self.z == o.z
        )

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


class Box2d(ToNumpy, Sequence):
    """2-D Bounding Box, defined by ``(xmin, ymin, xmax, ymax)``

    Attributes
    ----------
    xmin : float
        X-coordinate of the top-left point of the box.
    ymin : float
        Y-coordinate of the top-left point of the box.
    xmax : float
        X-coordinate of the bottm-right point of the box.
    ymax : float
        Y-coordinate of the bottm-right point of the box.

    Example
    -------

    >>> box = Box2d(1, 2, 3, 4)
    >>> box / 2
    Box2d(xmin=0.5, ymin=1.0, xmax=1.5, ymax=2.0)
    >>> box * (3.5, 5)
    Box2d(xmin=3.5, ymin=10.0, xmax=10.5, ymax=20.0)
    >>> # Box2d can be used directly with PIL.ImageDraw
    >>> draw = PIL.ImageDraw.Draw(img)
    >>> draw.rectangle(box, fill="green", width=2)
    """

    __UDT__ = Box2dType()

    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float):
        assert (
            0 <= xmin <= xmax
        ), f"xmin({xmin}) and xmax({xmax}) must satisfy 0 <= xmin <= xmax"
        assert (
            0 <= ymin <= ymax
        ), f"ymin({ymin}) and ymax({ymax}) must satisfy 0 <= ymin <= ymax"
        self.xmin = float(xmin)
        self.ymin = float(ymin)
        self.xmax = float(xmax)
        self.ymax = float(ymax)

    @classmethod
    def from_center(
        cls, center_x: float, center_y: float, width: float, height: float
    ) -> Box2d:
        """Factory method to construct a :py:class:`Box2d` from
        the center point coordinates: ``{center_x, center_y, width, height}``.


        Parameters
        ----------
        center_x : float
            X-coordinate of the center point of the box.
        center_y : float
            Y-coordinate of the center point of the box.
        width : float
            The width of the box.
        height : float
            The height of the box.

        Return
        ------
        Box2d
        """
        assert (
            width >= 0 and height >= 0
        ), f"Box2d width({width}) and height({height}) must be non-negative."
        return Box2d(
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        )

    @classmethod
    def from_top_left(
        cls, xmin: float, ymin: float, width: float, height: float
    ) -> Box2d:
        """Construct a :py:class:`Box2d` from
        the top-left based coordinates: ``{x0, y0, width, height}``.

        Top-left corner of an image / bbox is `(0, 0)`.

        Several public datasets, including `Coco Dataset`_, use this
        coordinations.

        Parameters
        ----------
        xmin : float
            X-coordinate of the top-left point of the box.
        ymin : float
            Y-coordinate of the top-left point of the box.
        width : float
            The width of the box.
        height : float
            The height of the box.


        References
        ----------
        - `Coco Dataset`_

        .. _Coco Dataset: https://cocodataset.org/
        """
        assert (
            width >= 0 and height >= 0
        ), f"Box2d width({width}) and height({height}) must be non-negative."
        return Box2d(xmin, ymin, xmin + width, ymin + height)

    def __repr__(self) -> str:
        return (
            f"Box2d(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}"
            + f", ymax={self.ymax})"
        )

    def __eq__(self, o: Box2d) -> bool:
        return isinstance(o, Box2d) and np.array_equal(
            self.to_numpy(), o.to_numpy()
        )

    def __len__(self) -> int:
        return 4

    def __getitem__(self, key: int) -> float:
        return [self.xmin, self.ymin, self.xmax, self.ymax][key]

    @staticmethod
    def _verified_scale(
        scale: Union[Real, Tuple[float, float]]
    ) -> Tuple[float, float]:
        if isinstance(scale, Real):
            assert scale > 0, f"scale must be positive, got {scale}"
            return scale, scale
        assert (
            type(scale) == tuple and len(scale) == 2
        ), f"scale must be either a number or a 2-element tuple, got {scale}"
        assert (
            scale[0] > 0 and scale[1] > 0
        ), f"scale must be positive, got {scale}"
        return scale

    def __truediv__(self, scale: Union[int, float, Tuple]) -> Box2d:
        """Scale down :py:class:`Box2d`

        Parameters
        ----------
        scale : number or a 2-element tuple/list
            Scale the Box2d by this amount.

        Return
        ------
        Box2d
            The scaled-down Box2d.

        """
        x_scale, y_scale = self._verified_scale(scale)
        return Box2d(
            xmin=self.xmin / x_scale,
            ymin=self.ymin / y_scale,
            xmax=self.xmax / x_scale,
            ymax=self.ymax / y_scale,
        )

    def __mul__(self, scale: Union[int, float, Tuple]) -> Box2d:
        """Scale up :py:class:`Box2d`

        Parameters
        ----------
        scale : number or a 2-element tuple/list
            Scale the Box2d by this amount.

        Return
        ------
        Box2d
            The scaled-up Box2d.
        """
        x_scale, y_scale = self._verified_scale(scale)
        return self / (1.0 / x_scale, 1.0 / y_scale)

    def to_numpy(self) -> np.ndarray:
        """Convert a :py:class:`Box2d` to numpy ndarray:
        ``array([xmin, ymin, xmax, ymax])``

        """
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax])

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

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
    """  # noqa: E501

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
        return (
            f"Box3d(center={self.center}, l={self.length}, "
            + f"h={self.height}, w={self.width}, heading={self.heading})"
        )

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
