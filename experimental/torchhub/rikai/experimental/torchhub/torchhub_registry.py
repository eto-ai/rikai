#  Copyright (c) 2022 Rikai Authors
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

from typing import Any
from urllib.parse import urlparse

import torch

from rikai.internal.reflection import find_func, has_func
from rikai.logging import logger
from rikai.spark.sql.codegen.base import Registry, udf_from_spec
from rikai.spark.sql.model import ModelSpec


class TorchHubModelSpec(ModelSpec):
    def __init__(self, repo_or_dir: str, model: str, raw_spec: ModelSpec):
        spec = {
            "version": "1.0",
            "schema": raw_spec["schema"],
            "model": {"flavor": raw_spec["flavor"], "uri": raw_spec["uri"]},
            "transforms": {
                "pre": raw_spec.get("preprocessor", None),
                "post": raw_spec.get("postprocessor", None),
            },
        }

        # remove none value of pre/post processing
        repo_proj = repo_or_dir.split(":")[0].replace("/", ".")
        pre_f = f"rikai.contrib.torchhub.{repo_proj}.{model}.pre_processing"
        post_f = f"rikai.contrib.torchhub.{repo_proj}.{model}.post_processing"
        schema_f = f"rikai.contrib.torchhub.{repo_proj}.{model}.OUTPUT_SCHEMA"

        if not spec["transforms"]["pre"]:
            if has_func(pre_f):
                spec["transforms"]["pre"] = pre_f
            else:
                del spec["transforms"]["pre"]
        if not spec["transforms"]["post"]:
            if has_func(post_f):
                spec["transforms"]["post"] = post_f
            else:
                del spec["transforms"]["post"]

        if not spec["schema"] and has_func(schema_f):
            spec["schema"] = find_func(schema_f)
        if not spec["model"]["flavor"]:
            spec["model"]["flavor"] = "pytorch"

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

    def resolve(self, raw_spec: dict):
        name = raw_spec["name"]
        uri = raw_spec["uri"]
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
        return udf_from_spec(spec)
