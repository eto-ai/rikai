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

from typing import Any, Optional
import random


class RandomShuffler:
    """Reservoir sampling-based shuffler to provide randomlized access over elements.

    :py:class:`RandomShuffler` maintains an internal buffer, and use `reservoir sampling`_
    to offer randomness with uniform distribution. Therefore, the buffer ``capacity`` does
    not affect the possibility distribution.

    Set ``capacity`` to ``1`` or ``0``, makes this :py:class:`RandomShuffler` a FIFO queue.

    Example
    -------

    .. code-block:: python

        def __iter__(self):
            \"\"\"Provide random access over a Stream\"\"\"
            shuffler = RandomShuffler(128)
            for elem in stream:
                shuffler.append(elem)
                # Explicit capacity control
                while shuffler.full():
                    yield shuffler.pop()
            while shuffler:
                yield shuffler.pop()

    References
    ----------
    - `Reservoir Sampling`_
    - Petastorm `Shuffling Buffer <https://github.com/uber/petastorm/blob/master/petastorm/reader_impl/shuffling_buffer.py>`_

    .. _Reservoir Sampling: https://en.wikipedia.org/wiki/Reservoir_sampling
    """

    def __init__(self, capacity: int, seed: Optional[int] = None):
        """Construct a :py:class:`RandomShuffler`

        Parameters
        ----------
        capacity : int
            The capacity of the internal random access buffer. Note that setting this value to
            1 or 0 makes this :py:class:`RandomShuffler` to a FIFO queue.
        seed : int, optional
            Random seed.
        """
        self.capacity = capacity
        self.seed = seed
        self.buffer = []
        random.seed(self.seed)

    def __repr__(self) -> str:
        return "RandomShuffler(capacity={})".format(self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def __bool__(self) -> bool:
        """Return True if this shuffler is not empty."""
        return len(self.buffer) > 0

    def full(self) -> bool:
        """Return True if this shuffler reaches to its capacity."""
        return len(self) >= self.capacity

    def append(self, elem: Any):
        """Append a new element to the shuffler"""
        self.buffer.append(elem)

    def pop(self) -> Any:
        """Pop out one random element from the buffer

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
