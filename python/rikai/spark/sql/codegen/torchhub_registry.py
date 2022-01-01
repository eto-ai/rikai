#  Copyright (c) 2021 Rikai Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Any, Dict, Optional
from urllib.parse import urlparse

import torch

from rikai.logging import logger
from rikai.spark.sql.codegen.base import ModelSpec, Registry, udf_from_spec


class TorchHubModelSpec(ModelSpec):
    def __init__(self, repo_or_dir: str, model: str, raw_spec: "ModelSpec"):
        spec = {
            "version": "1.0",
            "schema": raw_spec.getSchema(),
            "model": {
                "flavor": raw_spec.getFlavor(),
                "uri": raw_spec.getUri(),
            },
            "transforms": {
                "pre": raw_spec.getPreprocessor(),
                "post": raw_spec.getPostprocessor(),
            },
        }

        self.repo_or_dir = repo_or_dir
        self.model = model
        super().__init__(spec=spec, validate=True)

    def load_model(self) -> Any:
        """Load the model artifact specified in this spec"""
        return torch.hub.load(self.repo_or_dir, self.model)


class TorchHubRegistry(Registry):
    """TorchHub-based Model Registry"""

    def __repr__(self):
        return "TorchHubRegistry"

    def resolve(self, raw_spec):
        name = raw_spec.getName()
        uri = raw_spec.getUri()
        logger.info(f"Resolving model {name} from {uri}")
        parsed = urlparse(uri)
        if parsed.netloc:
            raise ValueError(
                "URI with 2 forward slashes is not supported, "
                "try URI with 1 slash instead"
            )
        if parsed.scheme != "torchhub":
            raise ValueError(
                f"Expect schema: torchhub, but got {parsed.scheme}"
            )
        parts = parsed.path.strip("/").split("/")
        if len(parts) != 3:
            raise ValueError("Bad URI, expected torchhub:///<org>/<prj>/<mdl>")
        repo_or_dir = "/".join(parts[:-1])
        model = parts[-1]
        spec = TorchHubModelSpec(repo_or_dir, model, raw_spec)
        udf = udf_from_spec(spec)
        return udf.func, udf.returnType
