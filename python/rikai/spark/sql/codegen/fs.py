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
from typing import Optional, Union
from urllib.parse import urlparse

import yaml

from rikai.io import open_uri
from rikai.logging import logger
from rikai.spark.sql.codegen.base import ModelSpec, Registry, udf_from_spec
from rikai.spark.sql.exceptions import SpecError

__all__ = ["FileSystemRegistry"]


class FileModelSpec(ModelSpec):
    """Model Spec.

    Parameters
    ----------
    spec_uri : str or Path
        Spec file URI
    options : Dict[str, Any], optional
        Additionally options. If the same option exists in spec already,
        it will be overridden.
    validate : bool, default True.
        Validate the spec during construction. Default ``True``.
    """

    def __init__(
        self,
        spec_uri: Union[str, Path],
        options: Optional[dict] = None,
        validate: bool = True,
    ):
        with open_uri(spec_uri) as fobj:
            spec = yaml.load(fobj, Loader=yaml.FullLoader)
        self.base_dir = os.path.dirname(spec_uri)
        spec.setdefault("options", {})
        if options:
            spec["options"].update(options)
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
        uri = raw_spec.get("uri")
        options = raw_spec.get("options", {})
        spec = FileModelSpec(uri, options=options)
        return spec
