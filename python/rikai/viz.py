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
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from rikai.conf import CONF_RIKAI_VIZ_COLOR, get_option
from rikai.mixin import Displayable, Drawable

__all__ = ["Style", "Text"]


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
        self.inner = None  # type: Optional[list[Drawable]]

    def __repr__(self):
        return f"style({self.kwargs})"

    def __call__(self, inner: Union[Drawable, list[Drawable]]) -> Drawable:
        # Make a copy of Style so the same style can be applied
        # to multiple drawables
        s = Style(**self.kwargs)
        if isinstance(inner, Drawable):
            inner = [inner]
        s.inner = inner
        return s

    def _render(self, render: Renderer, **kwargs):
        if self.inner is None:
            raise ValueError(
                "This style object has not attack to a Drawable yet"
            )
        # TODO: catch excessive parameters
        kwargs.update(self.kwargs)
        for inner_draw in self.inner:
            inner_draw._render(render, **kwargs)


class Draw(Displayable, ABC):
    """Draw is a container that contain the elements for visualized lazily."""

    def __init__(self):
        self.layers = []

    def __repr__(self):
        first_layer = self.layers[0] if self.layers else "N/A"
        return f"Draw({first_layer})"

    def _repr_mimebundle_(self, include=None, exclude=None):
        """default visualizer for embedded mime bundle"""
        return self.display()._repr_mimebundle_(
            include=include, exclude=exclude
        )

    def draw(self, layer: Union[Drawable, list[Drawable]]) -> Draw:
        # layer can not be checked against typing.Sequence or typing.Iterable,
        # because many of the Drawables are iterables (i.e., Box2d).
        if isinstance(layer, Drawable):
            layer = [layer]
        elif not isinstance(layer, (Drawable, list)):
            raise ValueError(
                f"{layer} must be one Drawable or a list of Drawable"
            )
        self.layers.extend(layer)
        return self

    def __or__(self, other: Union[Drawable, list[Drawable]]) -> Draw:
        return self.draw(other)


class Renderer(ABC):
    """The base class for rendering a :py:class:`Draw`."""

    @abstractmethod
    def rectangle(
        self, xy, color: str = get_option(CONF_RIKAI_VIZ_COLOR), width: int = 1
    ):
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


class PILRenderer(Renderer):
    """Use PIL to render drawables"""

    def __init__(self, img: PILImage):
        self.img = img.convert("RGBA")
        self.draw = ImageDraw.Draw(self.img)  # type: ImageDraw

    @property
    def image(self) -> PILImage:
        return self.img

    def rectangle(
        self, xy, color: str = get_option(CONF_RIKAI_VIZ_COLOR), width: int = 1
    ):
        self.draw.rectangle(xy, outline=color, width=width)

    def polygon(
        self,
        xy,
        color: str = get_option(CONF_RIKAI_VIZ_COLOR),
        fill: bool = True,
    ):
        if fill:
            overlay = PILImage.new("RGBA", self.img.size, (255, 255, 255, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            mask_img = PILImage.new("L", self.draw.im.size, 0)
            mask_draw = ImageDraw.Draw(mask_img)
            mask_draw.polygon(xy=xy, fill=color)
            overlay_draw.bitmap((0, 0), mask_img, fill=color)

            self.img = PILImage.alpha_composite(self.img, overlay)
            self.draw = ImageDraw.Draw(self.img)
        else:
            self.draw.polygon(xy=xy, outline=color)

    def text(
        self, xy, text: str, color: str = get_option(CONF_RIKAI_VIZ_COLOR)
    ):
        self.draw.text(xy, text, fill=color)

    def mask(
        self, arr: np.ndarray, color: str = get_option(CONF_RIKAI_VIZ_COLOR)
    ):
        overlay = PILImage.new("RGBA", self.img.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.bitmap((0, 0), PILImage.fromarray(arr), fill=color)

        self.img = PILImage.alpha_composite(self.img, overlay)
        self.draw = ImageDraw.Draw(self.img)


class Text(Drawable):
    """Render a Text

    Parameters
    ----------
    text : str
        The text content to be rendered
    xy : Tuple[int, int]
        The location to render the text
    color : str, optional
        The RGB color string to render the text
    """

    def __init__(
        self,
        text: str,
        xy: Tuple[int, int],
        color: str = get_option(CONF_RIKAI_VIZ_COLOR),
    ):
        self.text = text
        self.xy = xy
        self.color = color

    def _render(self, render: Renderer, **kwargs):
        kwargs["color"] = kwargs.get("color", self.color)
        return render.text(self.xy, self.text, **kwargs)
