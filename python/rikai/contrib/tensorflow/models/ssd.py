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

"""SSD: Single Shot MultiBox Detector

https://arxiv.org/abs/1512.02325
"""

from typing import Any, Callable, Dict

from rikai.types import Box2d

HUB_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"


def pre_processing(options: Dict[str, Any]):
    def pre_processor_func(batch):
        return batch

    return pre_processor_func


def post_processing(options: Dict[str, Any]) -> Callable:
    def post_process_func(batch):
        results = []
        for boxes, classes, scores in zip(
            batch["detection_boxes"].numpy(),
            batch["detection_classes"].numpy(),
            batch["detection_scores"].numpy(),
        ):
            predict_result = []
            for box, label_class, score in zip(boxes, classes, scores):
                predict_result.append(
                    {
                        "detection_boxes": Box2d(*box),
                        "detection_classes": int(label_class),
                        "detection_scores": float(score),
                    }
                )
            results.append(predict_result)
        return results

    return post_process_func


OUTPUT_SCHEMA = (
    "array<struct<detection_boxes:box2d, detection_scores:float,"
    + "detection_classes:int>>"
)
