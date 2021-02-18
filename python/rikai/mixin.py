#  Copyright 2020 Rikai Authors
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

"""Mixins
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Union

# Third Party
import numpy as np

# Rikai
from rikai.internal.uri_utils import uri_equal
from rikai.io import open_uri

__all__ = ["ToNumpy", "Asset", "Displayable"]


class ToNumpy(ABC):
    """ToNumpy Mixin."""

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        """Returns the content as a numpy ndarray."""


class Displayable(ABC):
    """Mixin for notebook viz"""

    @abstractmethod
    def display(self, **kwargs) -> "IPython.display.DisplayObject":
        """Return an IPython.display.DisplayObject"""


class Asset(ABC):
    """cloud asset Mixin.

    Rikai uses asset to store certain blob on the cloud storage, to facilitate
    the functionality like fast query, example inspections, and etc.

    An asset is also a cell in a DataFrame for analytics. It offers both fast
    query on columnar format and easy tooling to access the actual data.
    """

    def __init__(self, uri: Union[str, Path]) -> None:
        self.uri = str(uri)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Asset) and uri_equal(self.uri, o.uri)

    def open(self, mode="rb") -> BinaryIO:
        """Open the asset and returned as random-accessible file object."""
        return open_uri(self.uri, mode=mode)
