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

from typing import Optional

from rikai.spark.sql.codegen.base import ModelSpec, Registry
from rikai.spark.sql.codegen.mlflow_logger import MlflowLogger

__all__ = ["BootstrapRegistry"]


class BootstrapModelSpec(ModelSpec):
    """Bootstraped Model Spec.

    Parameters
    ----------
    options : Dict[str, Any], optional
        Additionally options. If the same option exists in spec already,
        it will be overridden.
    validate : bool, default True.
        Validate the spec during construction. Default ``True``.
    """

    def __init__(
        self,
        options: Optional[dict] = None,
        validate: bool = True,
    ):
        spec = self._load_spec_dict(options)
        super().__init__(spec, validate=validate)

    def _load_spec_dict(self, options: Optional[dict]) -> dict:
        """Convert the Run into a ModelSpec

        Parameters
        ----------
        options: dict, default None
            Runtime options to be used by the model/transforms

        Returns
        -------
        spec: Dict[str, Any]
        """
        spec = {
            "version": MlflowLogger._CURRENT_MODEL_SPEC_VERSION,
        }

        if options is not None and len(options) > 0:
            spec["options"] = options

        return spec


class BootstrapRegistry(Registry):
    """Bootrapped Model Registry"""

    def __repr__(self):
        return "BootstrapRegistry"

    def make_model_spec(self, raw_spec: dict):
        options = raw_spec.get("options", {})
        spec = BootstrapModelSpec(options=options)
        return spec
