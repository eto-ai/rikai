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

import requests
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
        label_fn: Optional[Callable[[int], str]] = None,
        collate_fn: Optional[Callable] = None,
        register: bool = True,
    ):
        """Initialize a TorchModelType

        Parameters
        ----------
        name : str
            The name of the model type
        pretrained_fn : Callable, optional
            The callable to be called if loading pretrained models.
        label_fn: Callable, optional
            Maps label_id to human-readable string label
        collate_fn : Callable, optional
            Customized collate fn to be called with PyTorch
            :py:class:`DataLoader`.
        register : bool
            Register the model to be discoverable via SQL
        """
        self.model: Optional[torch.nn.Module] = None
        self.spec: Optional[ModelSpec] = None
        # TODO: make this a class member?
        self.name = name
        self.pretrained_fn = pretrained_fn
        self.label_fn = label_fn
        self.collate_fn = collate_fn

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
            self.label_fn = self.spec.load_label_fn()
        self.model.eval()
        if "device" in kwargs:
            self.model.to(kwargs.get("device"))

    # Release GPU memory
    # https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/?ref=tfrecipes
    def release(self):
        model = self.model.cpu()
        del model
        torch.cuda.empty_cache()


_IMAGE_NET_CLASSES = []


def classification_label_fn(label_id):
    if not _IMAGE_NET_CLASSES:
        response = requests.get(
            "https://raw.githubusercontent.com/"
            "pytorch/hub/master/imagenet_classes.txt"
        )
        data = response.text
        _IMAGE_NET_CLASSES.extend(data.splitlines())
    return _IMAGE_NET_CLASSES[label_id]


class ClassificationModelType(TorchModelType):
    """Shared ModelType for image classification"""

    def __init__(
        self,
        name: str,
        pretrained_fn: Optional[Callable] = None,
        label_fn: Optional[Callable[[int], str]] = classification_label_fn,
        register: bool = True,
    ):
        super(ClassificationModelType, self).__init__(
            name,
            pretrained_fn=pretrained_fn,
            label_fn=label_fn,
            register=register,
        )

    def schema(self) -> str:
        return "struct<label_id:int, score:float, label:string>"

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
            if self.label_fn:
                r["label"] = self.label_fn(label)
            results.append(r)
        return results


# https://pytorch.org/vision/stable/models.html
COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def detection_label_fn(label_id: int) -> str:
    """Most pre-trained models are from Coco"""
    return COCO_INSTANCE_CATEGORY_NAMES[label_id]


def detection_collate_fn(batch: torch.Tensor) -> torch.Tensor:
    """TorchVision's models expect a list of `Tensor[C, H, W]`.

    https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
    """  # noqa: E501
    return batch


class ObjectDetectionModelType(TorchModelType):
    """Shared ModelType for object detections in Torchvision

    https://pytorch.org/vision/stable/models.html
    """

    def __init__(
        self,
        name: str,
        pretrained_fn: Optional[Callable] = None,
        label_fn: Optional[Callable[[int], str]] = detection_label_fn,
        collate_fn: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = detection_collate_fn,
        register: bool = True,
    ):
        super(ObjectDetectionModelType, self).__init__(
            name,
            pretrained_fn=pretrained_fn,
            label_fn=label_fn,
            collate_fn=collate_fn,
            register=register,
        )

    def __repr__(self):
        return f"ModelType({self.name})"

    def schema(self) -> str:
        return (
            "array<struct<box:box2d, score:float, label_id:int, label:string>>"
        )

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
                if self.label_fn:
                    r["label"] = self.label_fn(label)
                predict_result.append(r)
            results.append(predict_result)
        return results


# Registered model types
MODEL_TYPES = {}
