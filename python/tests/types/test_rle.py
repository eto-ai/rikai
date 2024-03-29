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

from rikai.types import rle


def test_rle_encoding():
    arr = np.asarray([0, 0, 0, 1, 1, 0, 0])
    assert np.array_equal(rle.encode(arr), [3, 2, 2])
    assert np.array_equal(rle.decode(rle.encode(arr), arr.shape), arr)

    arr = np.asarray([[0, 0, 0, 1, 1], [1, 0, 0, 1, 0]])
    assert np.array_equal(
        rle.encode(arr), [3, 3, 2, 1, 1]
    ), f"Rle result: {rle.encode(arr)}"
    assert np.array_equal(
        rle.decode(rle.encode(arr), shape=(2, 5), order="C"), arr
    ), f"Decoded: {rle.decode(rle.encode(arr), arr.shape)}"

    arr = np.asarray([1, 1, 0, 0])
    assert np.array_equal(rle.encode(arr), [0, 2, 2])
    assert np.array_equal(rle.decode(rle.encode(arr), arr.shape), arr)
