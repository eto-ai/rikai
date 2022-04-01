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

from enum import Enum
from numbers import Real
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

from rikai.mixin import Drawable, ToDict, ToNumpy
from rikai.spark.types.geometry import (
    Box2dType,
    Box3dType,
    MaskType,
    PointType,
)
from rikai.types import rle

__all__ = ["Point", "Box3d", "Box2d", "Mask"]


class Point(ToNumpy, ToDict):
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

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}


class Box2d(ToNumpy, Sequence, ToDict, Drawable):
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

    def _render(self, render: "rikai.viz.Renderer", **kwargs) -> None:
        render.rectangle(self, **kwargs)

    def to_numpy(self) -> np.ndarray:
        """Convert a :py:class:`Box2d` to numpy ndarray:
        ``array([xmin, ymin, xmax, ymax])``

        """
        return np.array([self.xmin, self.ymin, self.xmax, self.ymax])

    def to_dict(self) -> dict:
        return {
            "xmin": self.xmin,
            "ymin": self.ymin,
            "xmax": self.xmax,
            "ymax": self.ymax,
        }

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

    @staticmethod
    def _area(arr: np.ndarray) -> np.array:
        """Calculate box area from an array of boxes."""
        return np.maximum(0, arr[:, 2] - arr[:, 0]) * np.maximum(
            0, arr[:, 3] - arr[:, 1]
        )

    @staticmethod
    def ious(
        boxes1: Union[Sequence[Box2d], np.ndarray],
        boxes2: Union[Sequence[Box2d], np.ndarray],
    ) -> Optional[np.ndarray]:
        """Compute intersection over union(IOU).

        Parameters
        ----------
        boxes1 : :py:class:`numpy.ndarray`
            a list of Box2d with length of N
        boxes2 : :py:class:`numpy.ndarray`
            a list of Box2d with length of M

        Return
        ------
        :py:class:`numpy.ndarray`, optional
            For two lists of box2ds, which have the length of N, and M respectively,
            this function should return a N*M matrix, each element is the iou value
            `(float,[0, 1])`.
            Returns None if one of the input is empty.

        Example
        -------

        >>> import random
        >>>
        >>> def a_random_box2d():
        ...   x_min = random.uniform(0, 1)
        ...   y_min = random.uniform(0, 1)
        ...   x_max = random.uniform(x_min, 1)
        ...   y_max = random.uniform(y_min, 1)
        ...  return Box2d(x_min, y_min, x_max, y_max)
        >>>
        >>> list1 = [a_random_box2d() for _ in range(0, 2)]
        >>>
        >>> list2 = [a_random_box2d() for _ in range(0, 3)]
        >>>
        >>> Box2d.ious(list1, list2)
        """  # noqa: E501

        assert isinstance(boxes1, (Sequence, np.ndarray))
        assert isinstance(boxes2, (Sequence, np.ndarray))

        if not boxes1 or not boxes2:
            return None

        if not isinstance(boxes1, np.ndarray):
            boxes1 = np.array(boxes1)
        if not isinstance(boxes2, np.ndarray):
            boxes2 = np.array(boxes2)
        row_count = boxes1.shape[0]
        area1 = Box2d._area(boxes1).reshape(row_count, -1)
        area2 = Box2d._area(boxes2)

        xmin = np.maximum(boxes1[:, 0].reshape((row_count, -1)), boxes2[:, 0])
        ymin = np.maximum(boxes1[:, 1].reshape((row_count, -1)), boxes2[:, 1])
        xmax = np.minimum(boxes1[:, 2].reshape((row_count, -1)), boxes2[:, 2])
        ymax = np.minimum(boxes1[:, 3].reshape((row_count, -1)), boxes2[:, 3])

        inter_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        iou_mat = inter_area / (area1 + area2 - inter_area)

        return iou_mat

    def iou(
        self, other: Union[Box2d, Sequence[Box2d], np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Compute intersection over union(IOU)."""
        assert isinstance(
            other, (Box2d, Sequence, np.ndarray)
        ), f"Can only compute iou between Box2d, got {type(other)}"
        if isinstance(other, Box2d):
            other_arr = np.array([other])
        else:
            other_arr = np.array(other)
        if other_arr.size == 0:
            return np.zeros(0)

        self_arr = np.array(self)
        # Find intersection
        xmin = np.maximum(self_arr[0], other_arr[:, 0])
        ymin = np.maximum(self_arr[1], other_arr[:, 1])
        xmax = np.minimum(self_arr[2], other_arr[:, 2])
        ymax = np.minimum(self_arr[3], other_arr[:, 3])
        inter_arr = np.array([xmin, ymin, xmax, ymax]).T
        inter_area = self._area(inter_arr)

        iou_arr = inter_area / (
            self._area(np.array([self_arr]))
            + self._area(other_arr)
            - inter_area
        )
        if isinstance(other, Box2d):
            return iou_arr[0]
        return iou_arr


class Box3d(ToNumpy, ToDict):
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

    def to_dict(self) -> dict:
        return {
            "center": self.center,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "heading": self.heading,
        }


class Mask(ToNumpy, ToDict, Drawable):
    """2-d Mask over an image

    This 2D mask can be built from:

    - A binary-valued (0 or 1) 2D-numpy matrix.
    - A Run Length Encoded (RLE) data. It supports both row-based RLE
      (:py:class:`Mask.Type.RLE`) or column-based RLE
      (:py:class:`Mask.Type.COCO_RLE`) which is used in the Coco dataset.
    - A Polygon ``[x0, y0, x1, y1, ..., xn, yn]``

    Parameters
    ----------
    data: list or :py:class:`np.ndarray`
        The mask data. Can be a numpy array or a list.
    height: int, optional
        The height of the image this mask applies to.
    width: int, optional
        The width of the image this mask applies to.
    mask_type: :py:class:`Mask.Type`
        The type of the mask.

    Examples
    --------

    .. code-block:: python

        from pycocotools.coco import COCO
        from rikai.types import Mask

        coco = COCO("instance_train2017.json")
        ann = coco.loadAnns(ann_id)
        image = coco.loadImgs(ann["image_id"])
        if ann["iscrowed"] == 0:
            mask = Mask.from_polygon(
                ann["segmentation"],
                height=image["height"],
                width=image["width],
            )
        else:
            mask = Mask.from_coco_rle(
                ann["segmentation"]["counts"],
                height=image["height"],
                width=image["width],
            )

    """

    __UDT__ = MaskType()

    class Type(Enum):
        """Mask type."""

        POLYGON = 1
        RLE = 2
        COCO_RLE = 3  # COCO style RLE, column-based

    def __init__(
        self,
        data: Union[list, np.ndarray],
        height: Optional[int] = None,
        width: Optional[int] = None,
        mask_type: Mask.Type = Type.POLYGON,
    ):
        if mask_type != Mask.Type.POLYGON and (
            height is None or width is None
        ):
            raise ValueError("Must provide height and width for RLE type")

        self.type = mask_type
        self.data = data

        self.height = height
        self.width = width

    @staticmethod
    def from_rle(data: list[int], height: int, width: int) -> Mask:
        """Convert a (row-based) RLE mask (segmentation) into Mask

        Parameters
        ----------
        data: list[int]
            the RLE data
        height: int
            The height of the image which the mask applies to.
        width: int
            The width of the image which the mask applies to.
        """

        return Mask(data, height=height, width=width, mask_type=Mask.Type.RLE)

    @staticmethod
    def from_coco_rle(data: list[int], height: int, width: int) -> Mask:
        """Convert a COCO RLE mask (segmentation) into Mask

        Parameters
        ----------
        data: list[int]
            the RLE data
        height: int
            The height of the image which the mask applies to.
        width: int
            The width of the image which the mask applies to.
        """
        return Mask(
            data, height=height, width=width, mask_type=Mask.Type.COCO_RLE
        )

    @staticmethod
    def from_polygon(data: list[list[float]], height: int, width: int) -> Mask:
        """Build mask from a Polygon

        Parameters
        ----------
        data: list[list[float]]
            Multiple Polygon segmentation data. i.e.,
            ``[[x0, y0, x1, y1, ...], [x0, y0, x1, y1, ...]])``
        height: int
            The height of the image which the mask applies to.
        width: int
            The width of the image which the mask applies to.
        """
        return Mask(
            data, height=height, width=width, mask_type=Mask.Type.POLYGON
        )

    @staticmethod
    def from_mask(mask: np.ndarray) -> Mask:
        """Build mask from a numpy array.

        Parameters
        ----------
        mask : np.ndarray
            A binary-valued (0/1) numpy array
        """
        assert len(mask.shape) > 1, "Must have more than 2-dimensions"
        return Mask(data=rle.encode(mask), mask_type=Mask.Type.RLE)

    def __repr__(self):
        return f"Mask(type={self.type}, data=...)"

    def __eq__(self, other):
        return (
            isinstance(other, Mask)
            and self.type == other.type
            and self.height == other.height
            and self.width == other.width
            and np.array_equal(self.data, other.data)
        )

    def _polygon_to_mask(self) -> np.ndarray:
        arr = np.zeros((self.height, self.width), dtype=np.uint8)
        with Image.fromarray(arr) as im:
            draw = ImageDraw.Draw(im)
            for polygon in self.data:
                draw.polygon(list(np.array(polygon)), fill=1)
            return np.array(im)

    def _render(self, render, **kwargs):
        """Render a Mask"""
        if self.type == Mask.Type.POLYGON:
            for segmentation in self.data:
                render.polygon(segmentation, **kwargs)
        else:
            render.mask(self.to_mask())

    def to_mask(self) -> np.ndarray:
        """Convert this mask to a numpy array."""
        if self.type == Mask.Type.POLYGON:
            return self._polygon_to_mask()
        elif self.type == Mask.Type.RLE:
            return rle.decode(self.data, shape=(self.height, self.width))
        elif self.type == Mask.Type.COCO_RLE:
            return rle.decode(
                self.data, shape=(self.height, self.width), order="F"
            )
        else:
            raise ValueError("Unrecognized type")

    def to_numpy(self) -> np.ndarray:
        return self.to_mask()

    def to_dict(self) -> dict:
        ret = {
            "type": self.type.value,
            "width": self.width,
            "height": self.height,
            "data": self.data,
        }
        return ret

    def iou(self, other: Mask) -> float:
        this_mask = self.to_mask()
        other_mask = other.to_mask()
        intersection = np.count_nonzero(np.logical_and(this_mask, other_mask))
        union = np.count_nonzero(np.logical_or(this_mask, other_mask))
        try:
            return intersection / union
        except ZeroDivisionError:
            return 0
