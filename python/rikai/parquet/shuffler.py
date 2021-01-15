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

    See Also
    --------
    https://en.wikipedia.org/wiki/Reservoir_sampling
    """

    def __init__(self, capacity: int, seed: Optional[int] = None):
        """Construct a :py:class:`RandomShuffler`

        Parameters
        ----------
        capacity : int
            The capacity of the internal random access buffer. Note that if set this value to
            1 or 0 make this random shuffler to a FIFO queue.
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
        """Return True if it reaches to its capacity"""
        return len(self) >= self.capacity

    def append(self, elem: Any):
        """Append a new element to the shuffler"""
        self.buffer.append(elem)

    def pop(self) -> Any:
        if len(self.buffer) == 0:
            raise IndexError("Buffer is empty")
        idx = random.randrange(len(self.buffer))
        item = self.buffer[idx]
        self.buffer[idx] = self.buffer[-1]
        self.buffer.pop()
        return item
