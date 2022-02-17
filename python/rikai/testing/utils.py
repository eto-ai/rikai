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

from typing import Union, Iterator
from urllib.parse import urlparse

import pandas as pd

from rikai.spark.sql.codegen.base import codegen_from_spec
from rikai.spark.sql.codegen.fs import FileSystemRegistry
from rikai.spark.sql.model import ModelSpec


def _make_model_spec(raw_spec: "ModelSpec") -> ModelSpec:
    uri = raw_spec["uri"]
    parsed = urlparse(uri)
    scheme = parsed.scheme
    if scheme == "file":
        reg = FileSystemRegistry()
    elif scheme == "mlflow":
        from rikai.spark.sql.codegen.mlflow_registry import MlflowRegistry

        reg = MlflowRegistry()
    elif scheme == "torchhub":
        from rikai.experimental.torchhub.torchhub_registry import (
            TorchHubRegistry,
        )

        reg = TorchHubRegistry()
    elif scheme == "tfhub":
        from rikai.experimental.tfhub.tfhub_registry import TFHubRegistry

        reg = TFHubRegistry()
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    return reg.make_model_spec(raw_spec)


def _apply_model_func(func, inputs):
    results_list = [results for results in func(inputs)]
    return results_list


def _func_from_spec(spec: ModelSpec):
    codegen = codegen_from_spec(spec)
    return codegen.generate_inference_func(spec)


def apply_model_spec(
    spec: "ModelSpec",
    inputs: Union[Iterator[pd.Series], Iterator[pd.DataFrame]],
) -> Iterator[pd.Series]:
    if not isinstance(spec, ModelSpec):
        spec = _make_model_spec(spec)
    func = _func_from_spec(spec)
    return _apply_model_func(func, inputs)
