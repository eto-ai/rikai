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

"""Pytorch Dataset and DataLoader"""

import os
import uuid
from pathlib import Path
from typing import Callable, List, Union

# Third Party
import torch
from torch.utils.data import IterableDataset

# Rikai
import rikai.parquet
from rikai.pytorch.transforms import RikaiToTensor
from rikai.spark.utils import df_to_rikai

__all__ = ["Dataset"]


class Dataset(IterableDataset):
    """Rikai Pytorch Dataset.

    A :py:class:`torch.utils.data.IterableDataset` that reads
    Rikai data format. This :py:class:`Dataset` works with
    `multi-process data loading`_ using :py:class:`torch.utils.data.DataLoader`.

    Parameters
    ----------
    data_ref : str, Path, pyspark.sql.DataFrame
        URI to the data files or the dataframe
    columns : list of str, optional
        An optional list of column to load from parquet files.
    transform: Callable, default instance of RikaiToTensor
        Apply row level transformation before yielding each sample

    Note
    ----

    Up to ``pytorch==1.7``, :py:class:`~torch.utils.data.IterableDataset`
    does not work with :py:class:`torch.utils.data.Sampler` with
    :py:class:`torch.utils.data.DataLoader`.

    Use :py:class:`torch.utils.data.BufferedShuffleDataset` (torch>=1.8)
    with the Rikai dataset for randomness.

    Example
    -------

    >>> from rikai.pytorch.data import Dataset
    >>> from torch.utils.data import DataLoader
    >>>
    >>> dataset = Dataset("dataset", columns=["image", "label_id"])
    >>> # dataset = BufferedShuffleDataset(dataset)
    >>> loader = DataLoader(dataset, num_workers=8)

    .. _multi-process data loading: https://pytorch.org/docs/master/data.html#single-and-multi-process-data-loading
    """  # noqa: E501

    def __init__(
        self,
        data_ref: Union[str, Path, "pyspark.sql.DataFrame"],
        columns: List[str] = None,
        transform: Callable = RikaiToTensor(),
    ):
        super().__init__()
        self.data_ref = data_ref
        self.columns = columns
        self._transform = transform

    def __repr__(self) -> str:
        return f"Dataset(torch, {self.data_ref}, columns={self.columns})"

    def __iter__(self):
        rank = 0
        world_size = 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            rank = worker_info.id
            world_size = worker_info.num_workers

        uri = _maybe_cache_df(self.data_ref)
        for row in rikai.parquet.Dataset(
            uri,
            columns=self.columns,
            world_size=world_size,
            rank=rank,
        ):
            yield self._transform(row)


def _maybe_cache_df(
    data_ref: Union[str, Path, "pyspark.sql.DataFrame"]
) -> str:
    """
    If the given dataset_ref is a str/Path, then just return a str ref.
    If it's a pyspark DataFrame then cache the DataFrame as parquet and
    return the cache uri

    Parameters
    ----------
    data_ref: str, Path, or pyspark DataFrame
        Either a uri to parquet or a DataFrame that will be cached as parquet

    Returns
    -------
    uri: str
        Either the input parquet uri or the cached uri for a DataFrame
    """
    if isinstance(data_ref, (str, Path)):
        return str(data_ref)
    else:
        try:
            from pyspark.sql import DataFrame
        except ImportError:
            raise ImportError(
                "Cannot create rikai pytorch dataset from "
                "Spark DataFrame without pyspark installed."
            )
        if isinstance(data_ref, DataFrame):
            cache_uri = _get_cache_uri(data_ref)
            df_to_rikai(data_ref, cache_uri)
        else:
            raise TypeError(
                (
                    "dataset_ref must be a str, Path, or DataFrame and "
                    "not a {}."
                ).format(type(data_ref))
            )
        return cache_uri


def _get_cache_uri(df: "pyspark.sql.DataFrame") -> str:
    """
    TODO create a deterministic unique uri from the df for sharing

    Parameters
    ----------
    df: DataFrame
        The cache uri will be generated for the given DataFrame

    Returns
    -------
    cache_uri: str
        The uri to write the DataFrame to
    """
    cache_root_uri = rikai.options.rikai.cache_uri
    if not cache_root_uri:
        raise ValueError(
            "Could not retrieve rikai cache_uri from either "
            "spark or rikai configurations."
        )
    return os.path.join(cache_root_uri, str(uuid.uuid4()))
