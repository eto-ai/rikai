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

from __future__ import annotations

from abc import ABC
from typing import Callable

import numpy as np

from rikai.numpy import view
from rikai.spark.sql.model import ModelType, ModelSpec


class SklearnModelType(ModelType, ABC):
    def __init__(self):
        self.model = None

    def load_model(self, spec: ModelSpec, **kwargs):
        self.model = spec.load_model()

    def transform(self) -> Callable:
        # Do nothing in Sklearn
        pass


class Classification(SklearnModelType):
    """Classification model type"""

    def schema(self) -> str:
        return "struct<label_id: int>"

    def predict(self, x, *args, **kwargs) -> dict:
        assert self.model is not None
        return {"label_id": self.model.predict(x)[0]}


class Regression(SklearnModelType):
    def schema(self) -> str:
        return "float"

    def predict(self, x, *args, **kwargs) -> float:
        assert self.model is not None
        return self.model.predict(x)[0]


class DimensionalityReduction(SklearnModelType):
    def schema(self) -> str:
        return "ndarray"

    def predict(self, x, *args, **kwargs) -> np.ndarray:
        return view(self.model.predict(x))


MODEL_TYPES = {
    "linear_regression": Regression(),
    "random_forest_regression": Regression(),
    "logistic_regression": Classification(),
    "random_forest_classification": Classification(),
    "pca": DimensionalityReduction(),
}
