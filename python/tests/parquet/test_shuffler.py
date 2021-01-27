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


from rikai.parquet.shuffler import RandomShuffler


def shuffle_numbers(shuffler, numbers):
    returned = []
    for i in numbers:
        shuffler.append(i)
        while shuffler.full():
            returned.append(shuffler.pop())
    while shuffler:
        returned.append(shuffler.pop())
    return returned


def test_randomness():
    shuffler = RandomShuffler(16)
    expected = list(range(100))
    actual = shuffle_numbers(shuffler, expected)
    assert len(actual) == 100
    assert expected != actual
    assert expected == sorted(actual)


def test_randomness_with_large_capacity():
    """Test the case that the capacity is larger than total number
    of elements.
    """
    shuffler = RandomShuffler(128)
    expected = list(range(100))
    actual = shuffle_numbers(shuffler, expected)
    assert len(actual) == 100
    assert expected != actual
    assert expected == sorted(actual)


def test_fifo():
    shuffler = RandomShuffler(capacity=1)
    returned = shuffle_numbers(shuffler, range(100))
    assert len(returned) == 100
