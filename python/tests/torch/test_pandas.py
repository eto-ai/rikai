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


import pandas as pd
import numpy as np

from rikai.torch.pandas import PandasDataset


def test_pandas_dataframe():
    data = [{"id": i, "value": str(i * 10)} for i in range(10)]
    df = pd.DataFrame(data=data)

    dataset = PandasDataset(df)
    assert len(dataset) == 10
    print(list(dataset))
    assert list(dataset) == data


def test_pandas_series():
    data = list(range(10))
    series = pd.Series(data)

    dataset = PandasDataset(series)
    assert len(dataset) == 10
    assert np.array_equal(data, dataset)
