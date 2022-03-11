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

# Standard
from typing import Callable, Optional, Union

# Third Party
import pandas as pd
import tensorflow as tf

# Rikai
from rikai.parquet.dataset import convert_tensor
from rikai.spark.sql.codegen.base import unpickle_transform

__all__ = ["PandasDataset"]


class PandasDataset:
    """a Map-style Tensorflow dataset from a :py:class:`pandas.DataFrame` or a
    :py:class:`pandas.Series`.

    Note
    ----

    This class is used in Rikai's SQL-ML Spark implementation, which utilizes
    pandas UDF to run inference.
    """

    def __init__(
            self,
            df: Union[pd.DataFrame, pd.Series],
            transform: Optional[Callable] = None,
            unpickle: bool = False,
            use_pil: bool = False,
    ) -> None:
        assert isinstance(df, (pd.DataFrame, pd.Series))
        self.df = df
        self.transform = transform
        self.unpickle = unpickle
        self.use_pil = use_pil

    def data(self, batch_size):
        batch_size = 2
        print("batch_size:", batch_size)
        print("df type", type(self.df))
        print("df shape shape", self.df.shape)
        arr = self.df[0]
        print("arr type", arr)
        if self.unpickle:
            arr = unpickle_transform(arr)
        arr = convert_tensor(arr, use_pil=self.use_pil)
        import numpy as np

        from rikai.types.vision import Image

        img = Image.from_pil(arr).to_numpy()
        print("img shape", img.shape)
        data = tf.data.Dataset.from_tensors([img, img, img, img, img, img, img, img, img, img])
        # data = tf.data.Dataset.from_tensors(img)

        if self.transform:
            data.map(self.transform)
        # TODO batch seems not available yet
        data = data.as_numpy_iterator()
        return data
