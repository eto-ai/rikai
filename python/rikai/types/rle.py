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


import numpy as np
from typing import Tuple


def encode(arr: np.ndarray) -> np.ndarray:
    """Run-length encoding a matrix.

    """
    if len(arr.shape) > 1:
        arr = arr.reshape(-1)
    if len(arr) == 0:
        print(arr)
        return []
    total = len(arr)
    conti_idx = np.r_[0, np.flatnonzero(~np.equal(arr[1:], [arr[:-1]])) + 1]
    counts = np.diff(np.r_[conti_idx, total])
    return counts



def decode(rle: np.array, shape=Tuple[int]) -> np.ndarray:
    """Decode RLE encoding into a numpy mask."""

    val = 0

    pass
