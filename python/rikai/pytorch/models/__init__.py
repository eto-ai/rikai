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

"""Rikai-implemented PyTorch models and executors."""
from abc import ABC
from typing import Optional

import torch

from rikai.mixin import Pretrained
from rikai.spark.sql.model import ModelSpec, ModelType
from rikai.spark.sql.codegen.dummy import DummyModelSpec

__all__ = ["TorchModelType"]


class TorchModelType(ModelType, ABC):
    """Base ModelType for PyTorch models."""

    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.spec: Optional[ModelSpec] = None

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
