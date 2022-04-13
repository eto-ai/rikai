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
ResNet: Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
"""
import torchvision

from rikai.pytorch.models.torch import (
    classification_label_fn,
    ClassificationModelType,
)

resnet18 = ClassificationModelType(
    name="resnet18", pretrained_fn=torchvision.models.resnet18
)
resnet34 = ClassificationModelType(
    name="resnet34", pretrained_fn=torchvision.models.resnet34
)
resnet50 = ClassificationModelType(
    name="resnet50", pretrained_fn=torchvision.models.resnet50
)
resnet101 = ClassificationModelType(
    name="resnet101", pretrained_fn=torchvision.models.resnet101
)
resnet152 = ClassificationModelType(
    name="resnet152", pretrained_fn=torchvision.models.resnet152
)
# Make default resnet to be ResNet-50
resnet = ClassificationModelType(
    name="resnet", pretrained_fn=torchvision.models.resnet50
)
