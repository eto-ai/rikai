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

from rikai.pytorch.models.torch import ObjectDetectionModelType


def label_fn(label_id):
    # https://pytorch.org/vision/stable/models.html
    COCO_PERSON_KEYPOINT_NAMES = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    return COCO_PERSON_KEYPOINT_NAMES[label_id]


keypointrcnn = ObjectDetectionModelType(
    "keypointrcnn",
    pretrained_fn=torchvision.models.detection.keypointrcnn_resnet50_fpn,
    label_fn=label_fn,
)
