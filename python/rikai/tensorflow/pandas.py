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
from typing import Callable, Optional

# Third Party
import numpy as np
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
        df: pd.Series,
        transform: Optional[Callable] = None,
        unpickle: bool = False,
        use_pil: bool = False,
    ) -> None:
        assert isinstance(df, pd.Series)
        self.df = df
        self.transform = transform
        self.unpickle = unpickle
        self.use_pil = use_pil

    def batch(self, batch_size):
        def upickle_convent_transform(entity):
            if self.unpickle:
                entity = unpickle_transform(entity)
            entity = convert_tensor(entity, use_pil=self.use_pil)

            from rikai.types.vision import Image

            ret = Image.from_pil(entity).to_numpy()
            if self.transform:
                ret = self.transform(ret)
            return ret

        # TODO I want to use
        # `tensors = self.df.map(upickle_convent_transform).to_numpy()`
        # here, but it will cause unintelligible error
        tensors = np.array([upickle_convent_transform(x) for x in self.df])

        data = tf.data.Dataset.from_tensor_slices(tensors)
        # data = tf.data.Dataset.from_tensors(img)

        data = data.batch(batch_size).as_numpy_iterator()
        return data
