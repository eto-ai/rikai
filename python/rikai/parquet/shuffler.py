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

from abc import ABC, abstractmethod
from typing import Any, Optional
import random


class Shuffler(ABC):
    """Shuffler Abstract class"""

    @abstractmethod
    def full(self) -> bool:
        pass

    @abstractmethod
    def append(self, elem: Any):
        pass

    @abstractmethod
    def pop(self) -> Any:
        pass


class DummyShuffler(Shuffler):
    def __init__(self) -> None:
        super().__init__()
        self.buffer = []

    def __len__(self) -> int:
        return len(self.buffer)

    def __bool__(self) -> bool:
        return len(self.buffer) > 0

    def full(self) -> bool:
        return len(self.buffer) > 0

    def append(self, elem: Any):
        self.buffer.append(elem)

    def pop(self) -> Any:
        return self.buffer.pop()


class RandomShuffler(Shuffler):
    """

    Use reservoir sampling to randomlize the access elements.
    """

    def __init__(self, capacity: int, seed: Optional[int] = None):
        super().__init__()
        self.capacity = capacity
        self.seed = seed
        self.buffer = []
        random.seed(self.seed)

    def __repr__(self) -> str:
        return "RandomShuffler(capacity={})".format(self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def __bool__(self) -> bool:
        return len(self.buffer) > 0

    def full(self) -> bool:
        return len(self) >= self.capacity

    def append(self, elem: Any):
        self.buffer.append(elem)

    def pop(self) -> Any:
        if len(self) == 0:
            raise IndexError("Buffer is empty")
        idx = random.randrange(len(self))
        item = self.buffer[idx]
        self.buffer[idx] = self.buffer[-1]
        self.buffer.pop()
        return item
