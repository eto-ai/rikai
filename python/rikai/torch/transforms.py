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

from typing import Any, Mapping

import numpy as np
import pandas as pd

from rikai.mixin import ToNumpy, ToPIL

__all__ = ["RikaiToTensor"]


def convert_tensor(row, use_pil: bool = False):
    """Convert a parquet row into rikai semantic objects."""
    if not isinstance(row, (Mapping, pd.Series)):
        # Primitive values
        return row
    tensors = {}
    for key, value in row.items():
        if isinstance(value, dict):
            tensors[key] = convert_tensor(value)
        elif isinstance(value, (list, tuple)):
            tensors[key] = np.array([convert_tensor(elem) for elem in value])
        elif use_pil and isinstance(value, ToPIL):
            tensors[key] = value.to_pil()
        elif isinstance(value, ToNumpy):
            tensors[key] = value.to_numpy()
        else:
            tensors[key] = value
    return tensors


class RikaiToTensor:
    """Convert a Row in the ``Rikai`` parquet dataset into Pytorch Tensors

    Warnings
    --------
    Internal use only
    """

    def __init__(self, use_pil: bool = False):
        self.use_pil = use_pil

    def __repr__(self) -> str:
        return "ToTensor"

    def __call__(self, record) -> Any:
        return convert_tensor(record, use_pil=self.use_pil)
