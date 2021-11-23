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

import base64
import filecmp
from binascii import b2a_base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from rikai.types.geometry import Box2d
from rikai.types.vision import Image


def test_show_embedded_png(tmp_path):
    data = np.random.random((100, 100))
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    im = PILImage.fromarray(rescaled)
    uri = tmp_path / "test.png"
    im.save(uri)
    result = Image(uri)._repr_png_()
    with open(uri, "rb") as fh:
        expected = b2a_base64(fh.read()).decode("ascii")
        assert result == expected

        fh.seek(0)
        embedded_image = Image(fh)
        assert result == embedded_image._repr_png_()


def test_show_embedded_jpeg(tmp_path):
    data = np.random.random((100, 100))
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    im = PILImage.fromarray(rescaled)
    uri = tmp_path / "test.jpg"
    im.save(uri)
    result = Image(uri)._repr_jpeg_()
    with open(uri, "rb") as fh:
        expected = b2a_base64(fh.read()).decode("ascii")
        assert result == expected

        fh.seek(0)
        embedded_image = Image(fh)
        assert result == embedded_image._repr_jpeg_()


def test_format_kwargs(tmp_path):
    data = np.random.random((100, 100))
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    result_uri = tmp_path / "result.jpg"
    Image.from_array(rescaled, result_uri, format="jpeg", optimize=True)

    expected_uri = tmp_path / "expected.jpg"
    PILImage.fromarray(rescaled).save(
        expected_uri, format="jpeg", optimize=True
    )

    assert filecmp.cmp(result_uri, expected_uri)

    result_uri = tmp_path / "result.png"
    Image.from_array(rescaled, result_uri, format="png", compress_level=1)

    expected_uri = tmp_path / "expected.png"
    PILImage.fromarray(rescaled).save(
        expected_uri, format="png", compress_level=1
    )
    assert filecmp.cmp(result_uri, expected_uri)


def test_embeded_image_from_bytesio():
    data = np.random.random((100, 100))
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    im = PILImage.fromarray(rescaled)
    buf = BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    image = Image(buf)
    assert np.array_equal(image.to_numpy(), rescaled)


def test_crop_image():
    data = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    im = Image.from_array(data)
    patch = im.crop(Box2d(10, 10, 30, 30))
    cropped_data = patch.to_numpy()
    assert np.array_equal(cropped_data, data[10:30, 10:30])


def test_crop_real_image():
    uri = "http://farm2.staticflickr.com/1129/4726871278_4dd241a03a_z.jpg"
    img = Image(uri)
    data = img.to_numpy()
    patch = img.crop(Box2d(10, 10, 30, 30))
    assert np.array_equal(patch.to_numpy(), data[10:30, 10:30, :])


def test_crop_in_batch():
    uri = "http://farm2.staticflickr.com/1129/4726871278_4dd241a03a_z.jpg"
    img = Image(uri)
    data = img.to_numpy()
    patches = img.crop(
        [Box2d(10, 10, 30, 30), Box2d(15, 15, 35, 35), Box2d(20, 20, 40, 40)]
    )
    assert len(patches) == 3
    assert np.array_equal(patches[0].to_numpy(), data[10:30, 10:30, :])
    assert np.array_equal(patches[1].to_numpy(), data[15:35, 15:35, :])
    assert np.array_equal(patches[2].to_numpy(), data[20:40, 20:40, :])


@pytest.mark.timeout(30)
@pytest.mark.webtest
def test_show_remote_ref():
    from IPython.display import Image as IPyImage

    uri = "https://octodex.github.com/images/original.png"
    img = Image(uri)
    # TODO check the actual content
    assert img._repr_html_() == img.display()._repr_html_()
    assert img.display()._repr_html_() == IPyImage(url=uri)._repr_html_()


def test_save_image_as_external(tmp_path):
    # Store an embeded image to an external loczltion
    data = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    img = Image.from_array(data)
    ext_path = str(tmp_path / "ext.png")
    ext_img = img.save(ext_path)
    assert not ext_img.is_embedded
    assert ext_img.uri == ext_path
    assert Path(ext_path).exists()
    with img.to_pil() as img1, ext_img.to_pil() as img2:
        assert img1 == img2


def test_to_dict():
    data = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    img = Image.from_array(data)
    assert (
        base64.decodebytes(img.to_dict()["data"].encode("utf-8")) == img.data
    )
    img = Image("foo")
    assert img.to_dict() == {"uri": "foo"}
