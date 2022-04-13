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
    detection_label_fn,
    ObjectDetectionModelType,
)

ssd = ObjectDetectionModelType(
    "ssd", pretrained_fn=torchvision.models.detection.ssd.ssd300_vgg16
)

ssdlite = ObjectDetectionModelType(
    "ssdlite",
    pretrained_fn=torchvision.models.detection.ssdlite320_mobilenet_v3_large,
)
