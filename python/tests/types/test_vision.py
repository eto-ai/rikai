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
from typing import Union

import numpy as np
import pytest
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw

from rikai.io import open_uri
from rikai.types.geometry import Box2d
from rikai.types.vision import Image, ImageDraw
from rikai.viz import Style, Text


@pytest.fixture
def test_image() -> PILImage:
    data = np.random.random((100, 100, 3))
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    return PILImage.fromarray(rescaled)


def assert_mimebundle(obj: Union[Image, ImageDraw], prefix):
    mimebundle = obj.display()._repr_mimebundle_()
    assert len(mimebundle) == 1
    assert "text/html" in mimebundle
    assert mimebundle["text/html"].startswith(prefix)


def test_ipython_display(tmp_path, test_image: PILImage):
    data_uri_prefix = '<img src="data:image;base64,'
    box2d = Box2d(10, 10, 20, 20)

    # case 1: http uri
    project = "https://github.com/eto-ai/rikai"
    uri = f"{project}/raw/main/python/tests/assets/test_image.jpg"
    img1 = Image(uri)
    assert_mimebundle(img1, f'<img src="{uri}"/>')
    assert_mimebundle(img1 | box2d, data_uri_prefix)

    # case 2: non http uri
    uri = tmp_path / "test.jpg"
    test_image.save(uri)
    img2 = Image(uri)
    assert_mimebundle(img2, data_uri_prefix)
    assert_mimebundle(img2 | box2d, data_uri_prefix)

    # case 3: embeded image data
    img3 = Image(Image.read(uri).data)
    assert_mimebundle(img3, data_uri_prefix)
    assert_mimebundle(img3 | box2d, data_uri_prefix)


def test_convert_to_embedded_image(tmp_path, test_image: PILImage):
    uri = tmp_path / "test.jpg"
    test_image.save(uri)

    img = Image.read(uri)
    assert img.is_embedded
    with open(uri, mode="rb") as fobj:
        assert img.data == fobj.read()

    img2 = Image(uri)
    img3 = img2.to_embedded()
    assert not img2.is_embedded
    assert img3.is_embedded
    with open(uri, mode="rb") as fobj:
        assert img3.data == fobj.read()


def test_format_kwargs(tmp_path):
    data = np.random.random((100, 100, 3))
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
    data = np.random.random((100, 100, 3))
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    im = PILImage.fromarray(rescaled)
    buf = BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    image = Image(buf)
    assert np.array_equal(image.to_numpy(), rescaled)


def test_crop_image():
    data = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    im = Image.from_array(data)
    patch = im.crop(Box2d(10, 10, 30, 30))
    cropped_data = patch.to_numpy()
    assert np.array_equal(cropped_data, data[10:30, 10:30])


def test_crop_real_image(two_flickr_images: list):
    img = two_flickr_images[0]
    data = img.to_numpy()
    patch = img.crop(Box2d(10, 10, 30, 30))
    assert np.array_equal(patch.to_numpy(), data[10:30, 10:30, :])


def test_crop_in_batch(two_flickr_images: list):
    img = two_flickr_images[0]
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
    assert img.display()._repr_html_() == IPyImage(url=uri)._repr_html_()


def test_save_image_as_external(tmp_path):
    # Store an embedded image to an external location
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
    data = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    img = Image.from_array(data)
    assert (
        base64.decodebytes(img.to_dict()["data"].encode("utf-8")) == img.data
    )
    img = Image("foo")
    assert img.to_dict() == {"uri": "foo"}


def test_draw_image():
    data = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    img = Image.from_array(data)

    box1 = Box2d(1, 2, 10, 12)
    box2 = Box2d(20, 20, 40, 40)
    draw_boxes = img | box1 | box2
    pil_image = draw_boxes.to_image()

    expected = Image.from_array(data).to_pil()
    draw = PILImageDraw.Draw(expected)
    draw.rectangle((1.0, 2.0, 10.0, 12.0), outline="red")
    draw.rectangle((20, 20, 40, 40), outline="red")
    print(pil_image.to_numpy().shape, data.shape)
    assert np.array_equal(pil_image.to_numpy(), expected)


def test_draw_styled_images():
    data = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    img = Image.from_array(data)
    box1 = Box2d(1, 2, 10, 12)
    box2 = Box2d(20, 20, 40, 40)

    style = Style(color="yellow", width=3)
    styled_boxes = img | style(box1) | style(box2)

    expected = Image.from_array(data).to_pil()
    draw = PILImageDraw.Draw(expected)
    draw.rectangle((1, 2, 10, 12), outline="yellow", width=3)
    draw.rectangle((20, 20, 40, 40), outline="yellow", width=3)
    assert np.array_equal(styled_boxes.to_image().to_numpy(), expected)

    # Sugar!
    sugar_boxes = (
        img
        | box1 @ {"color": "green", "width": 10}
        | box2 @ {"color": "green", "width": 10}
    )

    sugar_expected = Image.from_array(data).to_pil()
    draw = PILImageDraw.Draw(sugar_expected)
    draw.rectangle((1, 2, 10, 12), outline="green", width=10)
    draw.rectangle((20, 20, 40, 40), outline="green", width=10)
    assert np.array_equal(sugar_boxes.to_image().to_numpy(), sugar_expected)


def test_draw_list_of_boxes():
    data = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    img = Image.from_array(data)
    boxes = [Box2d(1, 2, 10, 12), Box2d(20, 20, 30, 30)]
    pil_image = (img | boxes).to_image()

    expected = Image.from_array(data).to_pil()
    draw = PILImageDraw.Draw(expected)
    for box in boxes:
        draw.rectangle(box, outline="red")
    assert np.array_equal(pil_image.to_numpy(), expected)


def test_draw_styled_list_of_boxes():
    data = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    img = Image.from_array(data)
    boxes = [Box2d(1, 2, 10, 12), Box2d(20, 20, 30, 30)]
    style = Style(color="yellow", width=3)
    pil_image = (img | style(boxes)).to_image()

    expected = Image.from_array(data).to_pil()
    draw = PILImageDraw.Draw(expected)
    for box in boxes:
        draw.rectangle(box, outline="yellow", width=3)
    assert np.array_equal(pil_image.to_numpy(), expected)


def test_draw_texts():
    data = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    img = Image.from_array(data)
    pil_image = (img | Text("label", (10, 10))).to_image()

    expected = Image.from_array(data).to_pil()
    draw = PILImageDraw.Draw(expected)
    draw.text((10, 10), "label", fill="red")
    assert np.array_equal(pil_image.to_numpy(), expected)


def test_wrong_draw_order():
    """A draw pipeline must start with an image"""
    data = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    img = Image.from_array(data)

    box1 = Box2d(1, 2, 10, 12)

    with pytest.raises(TypeError):
        rendered = box1 | img

    with pytest.raises(TypeError):
        rendered = img | {"color": "white"} @ box1
