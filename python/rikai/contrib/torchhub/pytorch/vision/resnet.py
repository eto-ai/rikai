#  Copyright 2021 Rikai Authors
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

from typing import Any, Callable

import torch
from torchvision import transforms as T

from rikai.pytorch.models import TorchModelType

"""
Adapted from https://github.com/pytorch/pytorch.github.io/blob/site/assets/hub/pytorch_vision_resnet.ipynb
"""  # noqa E501


class ResnetModelType(TorchModelType):
    def __repr__(self):
        return "ResnetModelType"

    def schema(self) -> str:
        return "array<float>"

    def transform(self) -> Callable:
        """
        All pre-trained models expect input images normalized in the same way,
        i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where
        H and W are expected to be at least 224. The images have to be loaded
        in to a range of [0, 1] and then normalized using
        mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        """
        return T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
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
            results.append(result.detach().cpu().tolist())
        return results


MODEL_TYPE = ResnetModelType()
