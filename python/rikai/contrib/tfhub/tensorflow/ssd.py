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

from rikai.tensorflow.models import TensorflowModelType
from rikai.types.geometry import Box2d

HUB_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
TF_HUB_URL = "tfhub:///tensorflow/ssd_mobilenet_v2/2"


class SSDModelType(TensorflowModelType):
    def schema(self) -> str:
        return (
            "array<struct<detection_boxes:box2d,"
            "detection_scores:float, detection_classes:int>>"
        )

    def transform(self) -> Callable:
        return None

    def predict(self, images, *args, **kwargs) -> Any:
        # one image should have shape (height,width,3),
        # multiple image should have shape (number,height,width,3)
        assert len(images.shape) == 4
        assert (
            self.model is not None
        ), "model has not been initialized via load_model"
        results = []
        for image in images:
            # one image have the shape (x,x,3), but, input needs (1,x,x,3)
            # get more details from the doc here
            # https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2

            batch = self.model([image])

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


MODEL_TYPE = SSDModelType()
