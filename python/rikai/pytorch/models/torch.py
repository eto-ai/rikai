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
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from rikai.mixin import Pretrained
from rikai.spark.sql.codegen.dummy import DummyModelSpec
from rikai.spark.sql.model import ModelSpec, ModelType
from rikai.types import Box2d

__all__ = [
    "ObjectDetectionModelType",
    "TorchModelType",
    "ClassificationModelType",
    "MODEL_TYPES",
]

DEFAULT_MIN_SCORE = 0.5


class TorchModelType(ModelType, Pretrained, ABC):
    """Base ModelType for PyTorch models."""

    def __init__(
        self,
        name: str,
        pretrained_fn: Optional[Callable] = None,
        id_to_label_fn: Optional[Callable] = None,
        register: bool = True,
    ):
        """Initialize a TorchModelType

        Parameters
        ----------
        name : str
            The name of the model type
        pretrained_fn : Callable, optional
            The callable to be called if loading pretrained models.
        id_to_label_fn: Callable, optional
            Maps label_id to human readable string label
        register : bool
            Register the model to be discoverable via SQL
        """
        self.model: Optional[torch.nn.Module] = None
        self.spec: Optional[ModelSpec] = None
        # TODO: make this a class member?
        self.name = name
        self.pretrained_fn = pretrained_fn
        self.id_to_label_fn = id_to_label_fn

        if register:
            MODEL_TYPES[name] = self

    def __repr__(self):
        return f"ModelType({self.name})"

    def pretrained_model(self) -> Any:
        if self.pretrained_fn is None:
            raise ValueError(
                "Missing model URI. Not able to get pretrained model"
            )
        return self.pretrained_fn(pretrained=True)

    def load_model(self, spec: ModelSpec, **kwargs):
        self.spec = spec
        if isinstance(spec, DummyModelSpec):
            self.model = self.pretrained_model()
        else:
            self.model = self.spec.load_model()
            self.id_to_label_fn = self.spec.get_id_to_label_fn()
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
        return "struct<label_id: int, score: float>"

    def transform(self) -> Callable:
        return T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, images, *args, **kwargs) -> Any:
        assert (
            self.model is not None
        ), "model has not been initialized via load_model"
        batch = self.model(images)
        results = []
        for result in batch:
            scores = F.softmax(result, dim=0)
            label = torch.argmax(F.softmax(result, dim=0)).item()
            score = scores[label].item()
            r = {"label_id": label, "score": score}
            if self.id_to_label:
                r['label'] = self.id_to_label_fn(label)
            results.append(r)
        return results


class ObjectDetectionModelType(TorchModelType):
    """Shared ModelType for object detections in Torchvision

    https://pytorch.org/vision/stable/models.html
    """

    def __repr__(self):
        return f"ModelType({self.name})"

    def schema(self) -> str:
        return "array<struct<box:box2d, score:float, label_id:int>>"

    def transform(self) -> Callable:
        return T.ToTensor()

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
                r = {
                        "box": Box2d(*box),
                        "label_id": label,
                        "score": score,
                    }
                if self.id_to_label:
                    r['label'] = self.id_to_label_fn(label)
                predict_result.append(r)
            results.append(predict_result)
        return results


# Registered model types
MODEL_TYPES = {}
