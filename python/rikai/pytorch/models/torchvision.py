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

"""ModelSpecs for official torchvision models
"""

from typing import Any, Callable

from torchvision.transforms import ToTensor

from rikai.pytorch.models import TorchModelType
from rikai.types import Box2d

__all__ = ["ObjectDetectionModelType"]


DEFAULT_MIN_SCORE = 0.5


class ObjectDetectionModelType(TorchModelType):
    """Shared ModelSpec for object detections in Torchvision

    https://pytorch.org/vision/stable/models.html
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __repr__(self):
        return f"ModelType({self.name})"

    def schema(self) -> str:
        return "array<struct<box:box2d, score:float, label_id:int>>"

    def transform(self) -> Callable:
        return ToTensor()

    def predict(self, images, *args, **kwargs) -> Any:
        assert (
            self.model is not None
        ), "model has not been initialized via load_model"
        min_score = float(
            self.spec.options.get("min_score", DEFAULT_MIN_SCORE)
        )

        batch = self.model(images)
        results = []
        for predicts in batch:
            predict_result = []
            for box, label, score in zip(
                predicts["boxes"].tolist(),
                predicts["labels"].tolist(),
                predicts["scores"].tolist(),
            ):
                if score < min_score:
                    continue
                predict_result.append(
                    {
                        "box": Box2d(*box),
                        "label_id": label,
                        "score": score,
                    }
                )

            results.append(predict_result)
        return results
