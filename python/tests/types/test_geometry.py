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
from PIL import Image, ImageDraw

from rikai.types import Box2d


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
