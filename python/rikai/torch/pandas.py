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

# Standard
from typing import Any, Callable, Optional, Union

# Third Party
import pandas as pd
from torch.utils.data import Dataset

# Rikai
from rikai.torch.transforms import convert_tensor

__all__ = ["PandasDataset"]


class PandasDataset(Dataset):
    """a Map-style Pytorch dataset from a :py:class:`pandas.DataFrame` or a
    :py:class:`pandas.Series`.

    Note
    ----

    This class is used in Rikai's SQL-ML Spark implementation, which utilizes
    pandas UDF to run inference.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, pd.Series],
        transform: Optional[Callable] = None,
    ) -> None:
        assert isinstance(data, (pd.DataFrame, pd.Series))
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Any:
        row = self.data.iloc[index]
        row = convert_tensor(row)
        if self.transform:
            row = self.transform(row)
        return row
