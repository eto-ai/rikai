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

from typing import Any, Callable, Optional

import torch
import torchvision

from rikai.pytorch.models.torch import TorchModelType
from rikai.spark.sql.model import ModelSpec, parse_model_type

__all__ = ["FeatureExtractor", "FeatureExtractorType"]


class FeatureExtractor(torch.nn.Module):
    """Extract features"""

    def __init__(
        self,
        model: torch.nn.Module,
        node: str,
        output_field: str = "_rikai_out",
    ):
        super().__init__()
        self._node = node

        from torchvision.models.feature_extraction import (
            create_feature_extractor,
        )

        self.model = create_feature_extractor(model, {node: output_field})
        self._output_field = output_field

    def eval(self: FeatureExtractor) -> FeatureExtractor:
        self.model.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        preds = self.model(images)
        return preds[self._output_field]


class FeatureExtractorType(TorchModelType):
    """(Experimental) Generic Feature Extractor

    Examples
    --------

    .. code-block:: sql

        CREATE MODEL resnet_features
        FLAVOR pytorch
        MODEL_TYPE feature_extractors
        OPTIONS (model_type = 'resnet')  # Only resnet is tested
        USING '<uri to resnet model>'

        SELECT ML_PREDICT(resnet_features, image) AS embedding FROM images
    """

    def __init__(self):
        super().__init__("feature_extractor")
        self.original_model_type: Optional[TorchModelType] = None

    def load_model(self, spec: ModelSpec, **kwargs):
        self.spec = spec
        self.model = self.spec.load_model()
        self.original_model_type = parse_model_type(
            "pytorch", self.spec.options["model_type"]
        )
        self.model.eval()
        if "device" in kwargs:
            self.model.to(kwargs.get("device"))

    def schema(self) -> str:
        return "array<float>"

    def transform(self) -> Callable:
        assert (
            self.original_model_type is not None
        ), "The original model type has not been initialized"
        return self.original_model_type.transform()

    def predict(self, *args, **kwargs) -> Any:
        out = self.model(*args, **kwargs)
        batch = []
        for row in out.cpu():
            batch.append(row.T[0, 0].tolist())
        return batch


if torchvision.__version__ >= "0.11.0":
    feature_extractor = FeatureExtractorType()
