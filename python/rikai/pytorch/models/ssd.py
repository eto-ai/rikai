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

from rikai.mixin import Pretrained
from rikai.pytorch.models.torch import ObjectDetectionModelType, model_type


@model_type
class SSDModelType(ObjectDetectionModelType, Pretrained):
    def __init__(self):
        super().__init__("SSD")

    def pretrained_model(self):
        import torchvision
        return torchvision.models.detection.ssd.ssd300_vgg16(pretrained=True)


@model_type
class SSDLiteModelType(ObjectDetectionModelType, Pretrained):
    def __init__(self):
        super().__init__("ssdlite320_mobilenet_v3_large")

    def pretrained_model(self):
        import torchvision
        return torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            pretrained=True
        )
