#  Copyright 2021 Rikai Authors
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

import os
from pathlib import Path
from urllib.parse import urlparse

import yaml

from rikai.io import open_uri
from rikai.spark.sql.codegen.base import ModelSpec, Registry
from rikai.spark.sql.exceptions import SpecError

__all__ = ["FileSystemRegistry"]


class FileModelSpec(ModelSpec):
    """File-based Model Spec.

    Parameters
    ----------
    raw_spec : dict
        The RAW dict passed from SparkSession
    validate : bool, default True.
        Validate the spec during construction. Default ``True``.

    Raises
    ------
    SpecError
        If the spec is not correct.
    """

    def __init__(
        self,
        raw_spec: dict,
        validate: bool = True,
    ):
        spec = {
            "version": "1.0",
            "name": raw_spec.get("name"),
            "options": raw_spec.get("options", {}),
            "model": {
                "flavor": raw_spec.get("flavor"),
                "type": raw_spec.get("modelType"),
                "uri": raw_spec.get("uri"),
            },
        }
        uri = spec["model"]["uri"]
        if not uri:
            raise SpecError("Model URI is missing")
        if Path(uri).suffix in [".yml", ".yaml"]:
            with open_uri(uri) as fobj:
                spec = yaml.load(fobj, Loader=yaml.FullLoader)
            self.base_dir = os.path.dirname(uri)
        super().__init__(spec, validate=validate)

    def load_model(self):
        if self.flavor == "pytorch":
            from rikai.spark.sql.codegen.pytorch import load_model_from_uri

            return load_model_from_uri(self.model_uri)
        elif self.flavor == "tensorflow":
            from rikai.spark.sql.codegen.tensorflow import load_model_from_uri

            return load_model_from_uri(self.model_uri)
        else:
            raise SpecError("Unsupported flavor {}".format(self.flavor))

    @property
    def model_uri(self):
        """Absolute model URI."""
        origin_uri = super().model_uri
        parsed = urlparse(origin_uri)
        if parsed.scheme or os.path.isabs(origin_uri):
            return origin_uri
        return os.path.join(self.base_dir, origin_uri)


class FileSystemRegistry(Registry):
    """FileSystem-based Model Registry"""

    def __repr__(self):
        return "FileSystemRegistry"

    def make_model_spec(self, raw_spec: dict):
        return FileModelSpec(raw_spec)
