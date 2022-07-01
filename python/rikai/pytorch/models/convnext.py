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

import torchvision

from rikai.pytorch.models.torch import (
    ClassificationModelType,
)

if torchvision.__version__ >= "0.12.0":
    convnext = ClassificationModelType(
        "convnext", pretrained_fn=torchvision.models.convnext_base
    )
    convnext_base = ClassificationModelType(
        "convnext_base", pretrained_fn=torchvision.models.convnext_base
    )
    convnext_tiny = ClassificationModelType(
        "convnext_tiny", pretrained_fn=torchvision.models.convnext_tiny
    )
    convnext_small = ClassificationModelType(
        "convnext_small", pretrained_fn=torchvision.models.convnext_small
    )
    convnext_large = ClassificationModelType(
        "convnext_large", pretrained_fn=torchvision.models.convnext_large
    )
