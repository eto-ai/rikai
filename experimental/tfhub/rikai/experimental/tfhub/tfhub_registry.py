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

from rikai.internal.reflection import has_func
from rikai.logging import logger
from rikai.spark.sql.codegen.base import Registry
from rikai.spark.sql.model import ModelSpec
from rikai.spark.sql.model import is_fully_qualified_name


class TFHubModelSpec(ModelSpec):
    def __init__(self, publisher, handle, raw_spec: "ModelSpec"):
        spec = {
            "version": "1.0",
            "schema": raw_spec.get("schema", None),
            "model": {
                "flavor": raw_spec.get("flavor", None),
                "uri": raw_spec["uri"],
                "type": raw_spec.get("modelType", None),
            },
        }

        # remove None value
        if not spec["schema"]:
            del spec["schema"]
        # defaults to the `tensorflow` flavor
        if not spec["model"]["flavor"]:
            spec["model"]["flavor"] = "tensorflow"

        model_type = spec["model"]["type"]
        if model_type and not is_fully_qualified_name(model_type):
            model_type = f"rikai.contrib.tfhub.{publisher}.{model_type}"
            if has_func(model_type + ".MODEL_TYPE"):
                logger.info(f"tfhub model_type: {model_type}")
                spec["model"]["type"] = model_type

        self.handle = handle
        super().__init__(spec=spec, validate=True)

    def load_model(self) -> Any:
        """Load the model artifact specified in this spec"""
        return tfhub.load(self.handle)


class TFHubRegistry(Registry):
    """TFHub-based Model Registry"""

    def __repr__(self):
        return "TFHubRegistry"

    def make_model_spec(self, raw_spec: dict):
        uri = raw_spec["uri"]
        parsed = urlparse(uri)
        if parsed.netloc:
            raise ValueError(
                "URI with 2 forward slashes is not supported, "
                "try URI with 1 slash instead"
            )
        if parsed.scheme != "tfhub":
            raise ValueError(f"Expect schema: tfhub, but got {parsed.scheme}")
        handle = f"https://tfhub.dev{parsed.path}"
        publisher = parsed.path.lstrip("/").split("/")[0]
        spec = TFHubModelSpec(publisher, handle, raw_spec)
        return spec
