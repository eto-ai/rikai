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

from typing import Sequence

import numpy as np
import pytest
from PIL import Image, ImageDraw

from rikai.types import Box2d, Box3d, Mask, Point


def test_scale_box2d():
    box = Box2d(1.0, 2.0, 3.0, 4.0)

    for twos in [2, 2.0, np.float32(2), np.float64(2), (2, 2)]:
        assert Box2d(0.5, 1.0, 1.5, 2.0) == box / twos
        assert Box2d(2.0, 4.0, 6.0, 8.0) == box * twos

    assert Box2d(0.5, 0.5, 1.5, 1.0) == box / (2, 4)
    assert Box2d(0.5, 0.25, 1.5, 0.5) == box / (2.0, 8.0)
    assert Box2d(10.0, 15.0, 30.0, 30.0) == box * (10, 7.5)


def test_box2d_as_list():
    box = Box2d(1.0, 2.0, 3.0, 4.0)

    assert [1.0, 2.0, 3.0, 4.0] == list(box)

    img = Image.fromarray(
        np.random.randint(0, 128, size=(32, 32), dtype=np.uint8)
    )
    draw = ImageDraw.Draw(img)
    # Check that the box works with draw.
    draw.rectangle(box)

    assert isinstance(box, Sequence)


def test_box2d_iou():
    box1 = Box2d(0, 0, 20, 20)
    box2 = Box2d(10, 10, 30, 30)
    assert np.isclose(1 / 7, box1.iou(box2))
    assert isinstance(box1.iou(box2), float)
    box3 = Box2d(15, 15, 35, 35)
    assert np.isclose(5 * 5 / (2 * 20 * 20 - 5 * 5), box1.iou(box3))


def test_box2d_vectorize_iou():
    box1 = Box2d(0, 0, 20, 20)
    assert np.allclose(
        [1 / 7, 5 * 5 / (2 * 20 * 20 - 5 * 5)],
        box1.iou([Box2d(10, 10, 30, 30), Box2d(15, 15, 35, 35)]),
    )


def test_box2d_matrix_iou():
    vec1 = [Box2d(0, 0, 10, 10), Box2d(0, 0, 20, 20)]
    vec2 = [Box2d(5, 5, 10, 10), Box2d(10, 10, 20, 20)]
    assert np.allclose([[0.25, 0.0], [0.0625, 0.25]], Box2d.ious(vec1, vec2))


def test_box2d_empty_ious():
    boxes = [Box2d(0, 0, 10, 10), Box2d(0, 0, 20, 20)]
    assert Box2d.ious([], boxes) is None
    assert Box2d.ious(np.array([]), boxes) is None


def test_box2d_ious_bad_inputs():
    with pytest.raises(ValueError):
        Box2d.ious(None, None)

    boxes = [Box2d(0, 0, 10, 10), Box2d(0, 0, 20, 20)]
    with pytest.raises(ValueError):
        Box2d.ious(Box2d(1, 2, 3, 4), boxes)


def test_box2d_empty_iou():
    box1 = Box2d(0, 0, 20, 20)
    assert box1.iou([]).size == 0


def test_to_dict():
    b = Box2d(0, 0, 20, 20)
    exp = {"xmin": b.xmin, "ymin": b.ymin, "xmax": b.xmax, "ymax": b.ymax}
    assert b.to_dict() == exp

    p = Point(1, 2, 3)
    exp = {"x": 1, "y": 2, "z": 3}
    assert p.to_dict() == exp

    b3 = Box3d(1, 2, 3, 4, 5)
    exp = {
        "center": b3.center,
        "length": b3.length,
        "width": b3.width,
        "height": b3.height,
        "heading": b3.heading,
    }
    assert b3.to_dict() == exp


def test_mask_from_rle():
    mask = np.zeros((100, 100))
    assert np.array_equal(
        mask, Mask.from_rle([100 * 100], height=100, width=100).to_mask()
    )

    full_mask = np.ones((100, 100))
    assert np.array_equal(
        full_mask,
        Mask.from_rle([0, 100 * 100], height=100, width=100).to_mask(),
    )


def test_mask_to_dict():
    mask = Mask.from_polygon([[12, 23]], width=10, height=20)
    value = mask.to_dict()
    assert value["type"] == Mask.Type.POLYGON.value
    assert value["height"] == 20
    assert value["width"] == 10

    mask = Mask.from_rle([100 * 100], height=100, width=80)
    value = mask.to_dict()
    assert value["type"] == Mask.Type.RLE.value
    assert value["height"] == 100
    assert value["width"] == 80

    mask = Mask.from_coco_rle([100 * 100], height=100, width=80)
    value = mask.to_dict()
    assert value["type"] == Mask.Type.COCO_RLE.value
    assert value["height"] == 100
    assert value["width"] == 80
