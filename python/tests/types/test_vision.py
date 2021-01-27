#  Copyright 2021 Rikai Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import pytest

from binascii import b2a_base64
from PIL import Image as PILImage
import numpy as np

from rikai.types.vision import Image


def test_show_embedded_png(tmp_path):
    data = np.random.random((100, 100))
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    im = PILImage.fromarray(rescaled)
    uri = str(tmp_path / "test.png")
    im.save(uri)
    result = Image(uri)._repr_png_()
    with open(uri, "rb") as fh:
        expected = b2a_base64(fh.read()).decode("ascii")
        assert result == expected


def test_show_embedded_jpeg(tmp_path):
    data = np.random.random((100, 100))
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    im = PILImage.fromarray(rescaled)
    uri = str(tmp_path / "test.jpg")
    im.save(uri)
    result = Image(uri)._repr_jpeg_()
    with open(uri, "rb") as fh:
        expected = b2a_base64(fh.read()).decode("ascii")
        assert result == expected


@pytest.mark.webtest
def test_show_remote_ref():
    from IPython.display import Image as IPyImage

    uri = "https://octodex.github.com/images/original.png"
    img = Image(uri)
    # TODO check the actual content
    assert img._repr_html_() == img.display()._repr_html_()
    assert img.display()._repr_html_() == IPyImage(uri)._repr_html_()
