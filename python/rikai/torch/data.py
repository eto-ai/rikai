#  Copyright 2020 Rikai Authors
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
import threading
import uuid
import warnings
from concurrent.futures import as_completed, ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Dict, Generator, List, Optional, Union

# Third Party
import semver
import torch
from torch.utils.data import IterableDataset

import rikai.parquet

# Rikai
from rikai.conf import CONF_RIKAI_CACHEURI
from rikai.spark.utils import df_to_rikai
from rikai.torch.transforms import convert_tensor, RikaiToTensor

__all__ = ["DataLoader", "Dataset"]


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

    >>> from rikai.torch.data import Dataset
    >>> from torch.utils.data import DataLoader
    >>>
    >>> dataset = Dataset("dataset", columns=["image", "label"])
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


class DataLoader:
    """Rikai Dataset Loader in Pytorch.

    Parameters
    ----------
    data_ref : str
        The URI of a Rikai Dataset
    columns : list of str, optional
        An optional list of column to load from parquet files.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle the dataset or not. Default is False.
    num_workers : int
        The number of workers to download asset in parallel.
    seed : int, optional
        Provide random seed for shuffling process
    world_size : int
        The total number of distributed workers. Default is ``1``.
    rank : int
        The rank of this worker among distributed workers. Default is ``0``.

    **Distributed Training**

    :py:class:`DataLoader` can work with distributed training framework,
    such as `Horovod <https://horovod.readthedocs.io/en/stable/pytorch.html>`_.

    .. code-block:: python

        import torch
        import horovod.torch as hvd
        from rikai.torch.data import DataLoader

        # Initialize Horovod
        hvd.init()

        # Partition the dataset using Horovod primitives.
        train_loader = DataLoader(
            "s3://dataset/train",
            batch_size=16,
            world_size=hvd.size(),  # Horovod cluster size
            rank=hvd.rank())  # Local rank

        # Set ups on https://horovod.readthedocs.io/en/stable/pytorch.html

        for epoch in range(100):
            for batch_idx, (data, target) in enumerate(train_loader):
                ...

    .. warning::

        With Pytorch 1.8+, users should use the official :py:class:`torch.utils.data.DataLoader`
        with :py:class:`torch.utils.data.BufferedShuffleDataset` instead.

        This class will be deprecated later.

    References
    ----------
    .. `Horovod with Pytorch <https://horovod.readthedocs.io/en/stable/pytorch.html>`_

    """  # noqa

    def __init__(
        self,
        data_ref: Union[str, Path, "pyspark.sql.DataFrame"],
        columns: List[str] = None,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 16,
        collate_fn: Callable = None,
        seed: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
    ):  # pylint: disable=too-many-arguments
        assert isinstance(data_ref, (str, Path)) or (
            rank == 0 and world_size == 1
        ), "Only str/Path references are supported in distributed mode"
        data_ref = _maybe_cache_df(data_ref)
        self.dataset = rikai.parquet.Dataset(
            data_ref,
            columns=columns,
            shuffle=shuffle,
            seed=seed,
            world_size=world_size,
            rank=rank,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn if collate_fn else lambda x: x

        torch_version = semver.VersionInfo.parse(torch.__version__)
        if torch_version.major >= 1 and torch_version.minor >= 8:
            warnings.warn(
                "rikai.torch.data.DataLoader should be replaced with "
                "'torch.utils.data.BufferedShuffleDataset' and "
                "'torch.utils.data.DataLoader' "
                "in Pytorch 1.8+",
                DeprecationWarning,
            )

    def _prefetch(
        self, out_queue: Queue, done: threading.Event, stop: threading.Event
    ) -> None:
        """Prefetch dataset in a separate threads.

        Prefetch and convert the dataset to tensor, including download assets
        in the background.

        Parameters
        ----------
        loader : DataLoader
            The DataLoader object
        out_queue : Queue
            A thread-safe queue to collect converted rows.
        done : threading.Event
            Signal other threads that the prefetch is completed
        stop : threading.Event
            Signal from the calling thread to stop fetching data
        """
        try:

            def parallel_prefetch(executor, batch, q):
                futures = [executor.submit(convert_tensor, e) for e in batch]
                for fut in as_completed(futures):
                    q.put(fut.result())

            prefetch_batch = 4 * self.num_workers
            with ThreadPoolExecutor(self.num_workers) as executor:
                batch = []
                for example in self.dataset:
                    batch.append(example)
                    if len(batch) > prefetch_batch:
                        parallel_prefetch(executor, batch, out_queue)
                        batch = []
                    if stop.is_set():
                        break

                if batch:
                    parallel_prefetch(executor, batch, out_queue)
            out_queue.join()
        finally:
            # Signal the main thread that all datasat has been produced.
            done.set()

    def __iter__(self) -> Generator[Dict, None, None]:
        """Use DataLoader as a generator of example."""
        q = Queue(maxsize=self.num_workers * self.batch_size * 4)
        done = threading.Event()
        stop = threading.Event()

        prefetch_thd = threading.Thread(
            target=DataLoader._prefetch,
            args=(self, q, done, stop),
            daemon=True,
        )
        prefetch_thd.start()

        try:
            batch = []
            while not (q.empty() and done.is_set()):
                try:
                    batch.append(q.get(timeout=5))
                    q.task_done()
                    if len(batch) >= self.batch_size:
                        yield self.collate_fn(batch)
                        batch = []
                except Empty:
                    continue
            if batch:
                yield self.collate_fn(batch)
        finally:
            stop.set()  # tell the prefetch thread to stop
        prefetch_thd.join()
        q.join()


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
