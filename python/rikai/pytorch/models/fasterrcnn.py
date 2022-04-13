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

from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
)

from rikai.pytorch.models.torch import (
    detection_label_fn,
    ObjectDetectionModelType,
)

fasterrcnn = ObjectDetectionModelType(
    "fasterrcnn", pretrained_fn=fasterrcnn_resnet50_fpn
)

fasterrcnn_resnet50_fpn = ObjectDetectionModelType(
    "fasterrcnn_resnet50_fpn", pretrained_fn=fasterrcnn_resnet50_fpn
)

fasterrcnn_mobilenet_v3_large_fpn = ObjectDetectionModelType(
    "fasterrcnn_mobilenet_v3_large_fpn",
    pretrained_fn=fasterrcnn_mobilenet_v3_large_fpn,
)

fasterrcnn_mobilenet_large_320_fpn = ObjectDetectionModelType(
    "fasterrcnn_mobilenet_v3_large_320_fpn",
    pretrained_fn=fasterrcnn_mobilenet_v3_large_320_fpn,
)
