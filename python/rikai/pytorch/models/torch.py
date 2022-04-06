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

""":py:class:`ModelType` for official torchvision models
"""

from abc import ABC
from typing import Any, Callable, Optional, Type

import torch
from torchvision.transforms import ToTensor

from rikai.mixin import Pretrained
from rikai.spark.sql.codegen.dummy import DummyModelSpec
from rikai.spark.sql.model import ModelSpec, ModelType
from rikai.types import Box2d

__all__ = [
    "ObjectDetectionModelType",
    "TorchModelType",
    "ClassificationModelType",
    "MODEL_TYPES",
    "model_type"
]


DEFAULT_MIN_SCORE = 0.5


class TorchModelType(ModelType, ABC):
    """Base ModelType for PyTorch models."""

    def __init__(self, name: str):
        self.model: Optional[torch.nn.Module] = None
        self.spec: Optional[ModelSpec] = None
        # TODO: make this a class member?
        self.name = name

    def __repr__(self):
        return f"ModelType({self.name})"

    def load_model(self, spec: ModelSpec, **kwargs):
        self.spec = spec
        if isinstance(spec, DummyModelSpec):
            if isinstance(self, Pretrained):
                self.model = self.pretrained_model()
            else:
                raise ValueError("Missing model URI")
        else:
            self.model = self.spec.load_model()
        self.model.eval()
        if "device" in kwargs:
            self.model.to(kwargs.get("device"))

    # Release GPU memory
    # https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/?ref=tfrecipes
    def release(self):
        model = self.model.cpu()
        del model
        torch.cuda.empty_cache()


class ClassificationModelType(TorchModelType):
    """Shared ModelType for image classification"""

    def schema(self) -> str:
        return "array<float>"

    def transform(self) -> Callable:
        return ToTensor()

    def predict(self, images, *args, **kwargs) -> Any:
        assert (
            self.model is not None
        ), "model has not been initialized via load_model"
        return self.model(images)


class ObjectDetectionModelType(TorchModelType):
    """Shared ModelType for object detections in Torchvision

    https://pytorch.org/vision/stable/models.html
    """

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


# Registered model types
MODEL_TYPES = {}


def model_type(cls: Type[TorchModelType]) -> Type[TorchModelType]:
    """Decorator for registering a model type"""
    model = cls()
    if model.name in MODEL_TYPES:
        raise ValueError(f"Model {model.name} already registered")
    MODEL_TYPES[model.name] = model
    return cls
