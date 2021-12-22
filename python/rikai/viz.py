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
    """Styling a drawable-component.

    Examples
    --------

    >>> from rikai.viz import Style
    >>> from rikai.types import Box2d, Image
    ...
    >>> img = Image(uri="s3://....")
    >>> bbox1, bbox2 = Box2d(1, 2, 3, 4), Box2d(3, 4, 5, 6)
    >>> bbox_style = Style(color="yellow", width=4)
    >>> image | bbox_style(bbox1) | bbox_style(bbox2)
    """

    def __init__(self, **kwarg):
        self.kwargs = kwarg
        self.inner = None  # type: Optional[Drawable]

    def __repr__(self):
        return f"style({self.kwargs})"

    def __call__(self, inner: Drawable) -> Drawable:
        # Make a copy of Style so the same style can be applied
        # to multiple drawables
        s = Style(**self.kwargs)
        s.inner = inner
        return s

    def render(self, render: Render, **kwargs):
        assert self.inner is not None
        # TODO: catch excessive parameters
        return self.inner.render(render, **(self.kwargs | kwargs))


class Draw(Displayable, ABC):
    """Draw is a container that contain the elements for visualized lazily."""

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

    def __or__(self, other: Drawable) -> Draw:
        return self.draw(other)


class Render(ABC):
    """The base class for rendering a :py:class:`Draw`."""

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
    """Use PIL to render drawables"""

    def __init__(self, img: "PIL.Image"):
        from PIL import ImageDraw

        self.img = img.convert("RGBA")
        self.draw = ImageDraw.Draw(self.img)  # type: ImageDraw

    @property
    def image(self) -> "PIL.Image":
        return self.img

    def rectangle(self, xy, color: str = "red", width: int = 1):
        self.draw.rectangle(xy, outline=color, width=width)

    def polygon(self, xy, color: str = "red", fill_transparency: float = 0.2):
        self.draw.polygon(xy=xy, outline=color)
        if fill_transparency:
            from PIL import Image as PILImage, ImageDraw

            overlay = PILImage.new("RGBA", self.img.size, (255, 255, 255, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            mask_img = PILImage.new("L", self.draw.im.size, 0)
            mask_draw = ImageDraw.Draw(mask_img)
            mask_draw.polygon(xy=xy, fill=color)
            overlay_draw.bitmap((0, 0), mask_img, fill=color)

            self.img = PILImage.alpha_composite(self.img, overlay)
            self.draw = ImageDraw.Draw(self.img)

    def text(self, xy, text: str, color: str = ""):
        self.draw.text(xy, text, fill=color)

    def mask(self, arr: np.ndarray):
        pass
