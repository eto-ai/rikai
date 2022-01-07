#  Copyright 2022 Rikai Authors
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

from rikai.internal.reflection import check_class, check_func


def test_import_check():
    proj = "rikai.contrib.torchhub.pytorch.vision"
    assert check_class(f"{proj}.resnet.pre_processing")
    assert check_class(f"{proj}.resnet.post_processing")
    assert check_func(f"{proj}.resnet.pre_processing")
    assert check_func(f"{proj}.resnet.post_processing")

    assert not check_class(f"{proj}.resnet34.pre_processing")
    assert not check_class(f"{proj}.resnet34.post_processing")
    assert check_func(f"{proj}.resnet34.pre_processing")
    assert check_func(f"{proj}.resnet34.post_processing")

    assert not check_class("hello")
    assert not check_func("hello")
    assert not check_class("x.hello")
    assert not check_func("x.hello")
    assert not check_class("x.y.hello")
    assert not check_func("x.y.hello")
