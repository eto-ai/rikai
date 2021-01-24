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

import random
from typing import Generic, Optional, TypeVar

__all__ = ["RandomShuffler"]

Elem = TypeVar("Elem")


class RandomShuffler(Generic[Elem]):
    """Reservoir sampling-based shuffler to provide randomized access over elements.

    :py:class:`RandomShuffler` maintains an internal buffer, and uses `reservoir sampling`_
    to offer randomness with uniform distribution. The buffer ``capacity`` does
    not affect the possibility distribution.

    Parameters
    ----------
    capacity : int, optional
        The capacity of the internal random access buffer. Note that setting this value to
        ``1`` or ``0`` makes this :py:class:`RandomShuffler` to a FIFO queue. Default
        value: ``32``.
    seed : int, optional
        Random seed.

    Example
    -------

    .. code-block:: python

        def __iter__(self):
            \"\"\"Provide random access over a Stream\"\"\"
            shuffler = RandomShuffler(capacity=128)
            for elem in stream:
                shuffler.append(elem)
                # Approximately maintain the shuffler at its capacity.
                while shuffler.full():
                    yield shuffler.pop()
            while shuffler:
                yield shuffler.pop()

    Notes
    -----
    - Set ``capacity`` to ``1`` or ``0``, makes :py:class:`RandomShuffler` a FIFO queue.
    - This class is not thread-safe.

    References
    ----------
    - `Reservoir Sampling`_
    - Petastorm `Shuffling Buffer <https://github.com/uber/petastorm/blob/master/petastorm/reader_impl/shuffling_buffer.py>`_

    .. _Reservoir Sampling: https://en.wikipedia.org/wiki/Reservoir_sampling
    """  # noqa

    DEFAULT_CAPACITY = 32

    def __init__(
        self, capacity: int = DEFAULT_CAPACITY, seed: Optional[int] = None
    ):
        """Construct a :py:class:`RandomShuffler`"""
        self.capacity = capacity
        self.seed = seed
        self.buffer = []
        random.seed(self.seed)

    def __repr__(self) -> str:
        return "RandomShuffler(capacity={})".format(self.capacity)

    def __len__(self) -> int:
        """Returns the number of elements in the shuffler."""
        return len(self.buffer)

    def __bool__(self) -> bool:
        """Return True if this shuffler is not empty."""
        return len(self.buffer) > 0

    def full(self) -> bool:
        """Return True if this shuffler reaches to its capacity."""
        return len(self) >= self.capacity

    def append(self, elem: Elem):
        """Append a new element to the shuffler"""
        self.buffer.append(elem)

    def pop(self) -> Elem:
        """Pop out one random element from the shuffler.

        Raises
        ------
        IndexError
            If the internal buffer is empty.
        """
        if len(self.buffer) == 0:
            raise IndexError("Buffer is empty")
        idx = random.randrange(len(self.buffer))
        item = self.buffer[idx]
        self.buffer[idx] = self.buffer[-1]
        self.buffer.pop()
        return item
