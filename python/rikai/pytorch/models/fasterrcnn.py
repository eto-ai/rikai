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

from rikai.mixin import Pretrained
from rikai.pytorch.models.torch import model_type, ObjectDetectionModelType


@model_type
class FasterRCNNModelType(ObjectDetectionModelType, Pretrained):
    def __init__(self):
        super().__init__("fasterrcnn_resnet50_fpn")

    def pretrained_model(self):
        return torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )

@model_type
class FasterRCNNMobileNetV3ModelType(ObjectDetectionModelType, Pretrained):
    def __init__(self):
        super().__init__("fasterrcnn_mobilenet_v3_large_fpn")

    def pretrained_model(self):
        return torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            pretrained=True
        )


@model_type
class FasterRCNNMobileNetV3LargeModelType(
    ObjectDetectionModelType, Pretrained
):
    def __init__(self):
        super().__init__("fasterrcnn_mobilenet_v3_large_320_fpn")

    def pretrained_model(self):
        return (
            torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
                pretrained=True
            )
        )
