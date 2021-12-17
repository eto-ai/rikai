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

from __future__ import annotations

from typing import Tuple

import numpy as np


def encode(arr: np.ndarray) -> np.array:
    """Run-length encoding a matrix.

    Parameters
    ----------
    arr : a data array or n-D metrix/tensor.
    """
    if len(arr.shape) > 1:
        return encode(arr.reshape(-1))

    if len(arr) == 0:
        return []
    total = len(arr)
    conti_idx = np.r_[0, np.flatnonzero(~np.equal(arr[1:], [arr[:-1]])) + 1]
    counts = np.diff(np.r_[conti_idx, total])
    if arr[0]:
        counts = np.insert(counts, 0, 0)
    return counts


def decode(
    rle: np.array, shape: Tuple[int] | Tuple[int, int], order: str = "C"
) -> np.ndarray:
    """Decode (COCO) RLE encoding into a numpy mask.

    Parameters
    ----------
    rle : np.array
        A 1-D array of RLE encoded data.
    shape: tuple of ints
        (height, width)
    order: str
        Numpy array order. If uses Coco-style RLE, order should set to F.
    """
    val = 0
    start_idx = 0
    n = np.sum(rle)
    arr = np.full(n, 0, dtype=np.uint8)
    for length in rle:
        arr[start_idx: start_idx + length] = val  # fmt:skip
        start_idx += length
        val = 1 - val

    return arr.reshape(shape, order=order)
