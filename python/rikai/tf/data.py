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

"""Tensorflow Dataset"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import tensorflow as tf

from rikai.parquet import Dataset
from rikai.parquet.dataset import convert_tensor

__all__ = ["from_rikai"]


def from_rikai(
    data_ref: Union[str, Path],
    output_signature: Tuple,
    columns: Optional[List[str]] = None,
):
    """Build a Tensorflow (tf) Dataset from rikai dataset.

    Parameters
    ----------
    query : str
        A dataset URI
    output_signature : tuple
        A (nested) structure of `tf.TypeSpec` objects corresponding
        to each component.
    columns : List[str], optional
        To read only given columns

    """

    def dataset_generator():
        dataset = Dataset(data_ref)
        for item in dataset:
            yield tuple(convert_tensor(item).values())

    return tf.data.Dataset.from_generator(
        dataset_generator, output_signature=output_signature
    )
