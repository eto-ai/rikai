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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Empty, Queue
from typing import Callable, Dict, Generator, List, Optional

# Third Party
import numpy as np

# Rikai
from rikai.mixin import ToNumpy
from rikai.parquet.dataset import Dataset

__all__ = ["DataLoader"]


class DataLoader:
    """Rikai Dataset Loader in Pytorch.

    Parameters
    ----------
    dataset : str
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

    References
    ----------
    .. `Horovod with Pytorch <https://horovod.readthedocs.io/en/stable/pytorch.html>`_

    """  # noqa

    def __init__(
        self,
        dataset: str,
        columns: List[str] = None,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 16,
        collate_fn: Callable = None,
        seed: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
    ):  # pylint: disable=too-many-arguments
        self.dataset = Dataset(
            dataset,
            columns=columns,
            shuffle=shuffle,
            seed=seed,
            world_size=world_size,
            rank=rank,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn if collate_fn else lambda x: x

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


def convert_tensor(row):
    """Convert a parquet row into rikai semantic objects."""
    tensors = {}
    for key, value in row.items():
        if isinstance(value, dict):
            tensors[key] = convert_tensor(value)
        elif isinstance(value, (list, tuple)):
            tensors[key] = np.array([convert_tensor(elem) for elem in value])
        elif isinstance(value, ToNumpy):
            tensors[key] = value.to_numpy()
        else:
            tensors[key] = value
    return tensors
