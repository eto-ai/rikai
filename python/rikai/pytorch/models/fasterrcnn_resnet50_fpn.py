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

from typing import Any, Callable, Optional

import torch
from torchvision.transforms import ToTensor

from rikai.spark.sql.model import ModelSpec, SpecPayload
from rikai.types import Box2d

__all__ = ["SPEC"]


DEFAULT_MIN_SCORE = 0.5


class FasterRCNNSpec(ModelSpec):
    def __init__(self):
        super().__init__()
        self.model: Optional[torch.nn.Module] = None
        self.spec: Optional[SpecPayload] = None

    def schema(self) -> str:
        return "array<struct<box:box2d, score:float, label:int>>"

    def load_model(self, raw_spec: SpecPayload, device=None):
        self.model = raw_spec.load_model()
        self.model.eval()
        if device:
            self.model = self.model.to(device)
        self.spec = raw_spec
        return self

    def transform(self) -> Callable:
        return ToTensor()

    def predict(self, images, *args, **kwargs) -> Any:
        assert (
            self.model is not None
        ), "model is not initialized via load_model"
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
                        "label": label,
                        "score": score,
                    }
                )

            results.append(predict_result)
        return results

    def release(self):
        model = self.model.cpu()
        del model
        torch.cuda.empty_cache()


SPEC = FasterRCNNSpec()
