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

from abc import ABC, abstractmethod
from typing import Any, Callable

from rikai.mixin import ToDict
from rikai.spark.sql.codegen.base import ModelSpec


class Spec(ToDict, ABC):
    @abstractmethod
    def schema(self) -> str:
        pass

    @abstractmethod
    def load_model(self, raw_spec: ModelSpec):
        pass

    @abstractmethod
    def transform(self) -> Callable:
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        pass

    def __call__(self, *args, **kwargs) -> Any:
        return self.predict(*args, **kwargs)
