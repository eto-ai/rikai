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

"""
EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
https://arxiv.org/abs/1905.11946
"""

import torchvision

from rikai.pytorch.models.torch import (
    ClassificationModelType,
)

if torchvision.__version__ >= "0.11.0":
    efficientnet_b0 = ClassificationModelType(
        "efficientnet_b0", pretrained_fn=torchvision.models.efficientnet_b0
    )
    efficientnet_b1 = ClassificationModelType(
        "efficientnet_b1", pretrained_fn=torchvision.models.efficientnet_b1
    )
    efficientnet_b2 = ClassificationModelType(
        "efficientnet_b2", pretrained_fn=torchvision.models.efficientnet_b2
    )
    efficientnet_b3 = ClassificationModelType(
        "efficientnet_b3", pretrained_fn=torchvision.models.efficientnet_b3
    )
    efficientnet_b4 = ClassificationModelType(
        "efficientnet_b4", pretrained_fn=torchvision.models.efficientnet_b4
    )
    efficientnet_b5 = ClassificationModelType(
        "efficientnet_b5", pretrained_fn=torchvision.models.efficientnet_b5
    )
    efficientnet_b6 = ClassificationModelType(
        "efficientnet_b6", pretrained_fn=torchvision.models.efficientnet_b6
    )
    efficientnet_b7 = ClassificationModelType(
        "efficientnet_b7", pretrained_fn=torchvision.models.efficientnet_b7
    )
