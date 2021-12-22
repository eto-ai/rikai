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

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from rikai.mixin import Displayable, Drawable

__all__ = ["Style"]


class Style(Drawable):
    """Styling a drawable-component."""
    def __init__(self, **kwarg):
        self.kwargs = kwarg
        self.inner = None  # type: Optional[Drawable]

    def __repr__(self):
        return f"style({self.kwargs})"

    def __call__(self, inner: Drawable) -> Drawable:
        s = Style(**self.kwargs)
        s.inner = inner
        return s

    def render(self, render: Render, **kwargs):
        assert self.inner is not None
        # TODO: catch excessive parameters
        return self.inner.render(render, **(self.kwargs | kwargs))


class Draw(Displayable, ABC):
    """Draw is a container that contain the elements for visualized lazily.
    """

    def __init__(self):
        self.layers = []

    def __repr__(self):
        first_layer = self.layers[0] if self.layers else "N/A"
        return f"Draw({first_layer})"

    def draw(self, layer: Drawable) -> Draw:
        if not isinstance(layer, Drawable):
            raise ValueError(f"{layer} must be a Drawable")
        self.layers.append(layer)
        return self

    def __and__(self, other: Drawable) -> Draw:
        return self.draw(other)


class Render(ABC):
    @abstractmethod
    def rectangle(self, xy, color: str = "red", width: int = 1):
        pass

    @abstractmethod
    def polygon(self, xy):
        pass

    @abstractmethod
    def text(self, xy, text: str, color: str = ""):
        pass

    @abstractmethod
    def mask(self, arr: np.ndarray):
        pass


class PILRender(Render):
    """Use PIL to render components"""

    def __init__(self, draw: "PIL.ImageDraw"):
        from PIL import ImageDraw
        self.draw = draw  # type: ImageDraw

    def rectangle(self, xy, color: str = "red", width: int = 1):
        self.draw.rectangle(xy, outline=color, width=width)

    def polygon(self, xy):
        pass

    def text(self, xy, text: str, color: str = ""):
        self.draw.text(xy, text, fill=color)

    def mask(self, arr: np.ndarray):
        pass
