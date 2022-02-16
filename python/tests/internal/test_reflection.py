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

from rikai.internal.reflection import has_func


def test_import_check():
    proj = "rikai.contrib.torchhub.pytorch.vision"
    # Check usage:
    # from {proj} import resnet; resnet.MODEL_TYPE
    assert has_func(f"{proj}.resnet.MODEL_TYPE")

    # Check usage:
    # from {proj}.resnet34 import pre_processing
    assert has_func(f"{proj}.resnet34.MODEL_TYPE")
    assert has_func(f"{proj}.resnet18.MODEL_TYPE")

    # Negative usages
    assert not has_func("hello")
    assert not has_func("x.hello")
    assert not has_func("x.y.hello")
