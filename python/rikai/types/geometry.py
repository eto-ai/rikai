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
        self.x = x
        self.y = y
        self.z = z

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])