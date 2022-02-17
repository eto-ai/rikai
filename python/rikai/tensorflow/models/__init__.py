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

"""Rikai-implemented Tensorflow models and executors."""
from abc import ABC
from typing import Optional

from rikai.spark.sql.model import ModelSpec, ModelType

__all__ = ["TensorflowModelType"]


class TensorflowModelType(ModelType, ABC):
    """Base ModelType for Tensorflow models."""

    def __init__(self):
        self.model = None
        self.spec: Optional[ModelSpec] = None

    def load_model(self, spec: ModelSpec, **kwargs):
        self.model = spec.load_model()
        self.spec = spec
