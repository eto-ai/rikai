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

import tensorflow_hub as tfhub

from rikai.internal.reflection import find_func, has_func
from rikai.logging import logger
from rikai.spark.sql.codegen.base import Registry, udf_from_spec
from rikai.spark.sql.model import ModelSpec


class TFHubModelSpec(ModelSpec):
    def __init__(self, handle, raw_spec: ModelSpec):
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
        if not spec["transforms"]["pre"]:
            del spec["transforms"]["pre"]
        if not spec["transforms"]["post"]:
            del spec["transforms"]["post"]

        # defaults to the `tensorflow` flavor
        if not spec["model"]["flavor"]:
            spec["model"]["flavor"] = "tensorflow"

        self.handle = handle
        super().__init__(spec=spec, validate=True)

    def load_model(self) -> Any:
        """Load the model artifact specified in this spec"""
        return tfhub.load(self.handle)


class TFHubRegistry(Registry):
    """TFHub-based Model Registry"""

    def __repr__(self):
        return "TFHubRegistry"

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
        if parsed.scheme != "tfhub":
            raise ValueError(f"Expect schema: tfhub, but got {parsed.scheme}")
        handle = f"https://tfhub.dev{parsed.path}"
        spec = TFHubModelSpec(handle, raw_spec)
        return udf_from_spec(spec)
