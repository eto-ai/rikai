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


import numpy as np
import pandas as pd

from rikai.torch.pandas import PandasDataset
from rikai.types.geometry import Box2d


def test_pandas_dataframe():
    data = [{"id": i, "value": str(i * 10)} for i in range(10)]
    df = pd.DataFrame(data)

    dataset = PandasDataset(df)
    assert len(dataset) == 10
    assert [d["data"] for d in dataset] == data


def test_dataframe_with_udt():
    data = [{"id": i, "box": Box2d(i, i + 1, i + 2, i + 3)} for i in range(10)]
    df = pd.DataFrame(data)

    dataset = PandasDataset(df)
    assert len(dataset) == 10
    expected = [
        {"id": i, "box": np.array([i, i + 1, i + 2, i + 3])} for i in range(10)
    ]
    for i in range(10):
        row = dataset[i]
        assert row["data"]["id"] == expected[i]["id"]
        assert np.array_equal(row["data"]["box"], expected[i]["box"])


def test_pandas_series():
    data = list(range(10))
    series = pd.Series(data)

    dataset = PandasDataset(series)
    assert len(dataset) == 10
    assert np.array_equal(data, [d["data"] for d in dataset])


def test_pandas_series_with_udt():
    data = pd.Series([Box2d(i, i + 1, i + 2, i + 3) for i in range(10)])
    dataset = PandasDataset(data)
    assert len(dataset) == 10
    for i in range(10):
        assert np.array_equal(
            dataset[i]["data"], np.array([i, i + 1, i + 2, i + 3])
        )
