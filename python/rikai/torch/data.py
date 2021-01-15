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
    """Rikai Dataset Loader in Pytorch."""

    def __init__(
        self,
        dataset: str,
        columns: List[str] = None,
        batch_size: int = 1,
        num_workers: int = 16,
        shuffle: bool = False,
        transform_fn: Callable = lambda x: (x,),
        collate_fn: Callable = None,
        shuffle_pool_size: int = 2 ** 10,
        seed: Optional[int] = None,
    ):  # pylint: disable=too-many-arguments
        self.dataset = Dataset(
            dataset,
            columns=columns,
            shuffle=shuffle,
            shuffle_pool_size=shuffle_pool_size,
            seed=seed,
            world_size=world_size,
            rank=rank,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn if collate_fn else lambda x: x
        self.transform_fn = transform_fn

    @staticmethod
    def _convert_tensor(row):
        """Convert a parquet row into rikai semantic objects."""
        tensors = {}
        for key, value in row.items():
            if isinstance(value, dict):
                tensors[key] = DataLoader._convert_tensor(value)
            elif isinstance(value, (list,)):
                tensors[key] = np.array(
                    [DataLoader._convert_tensor(elem) for elem in value]
                )
            elif isinstance(value, ToNumpy):
                tensors[key] = value.to_numpy()
            else:
                tensors[key] = value
        return tensors

    @staticmethod
    def prefetch(loader: "DataLoader", out_queue: Queue, done: threading.Event) -> None:
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
            The signal of prefetch is done.
        """

        def parallel_prefetch(executor, batch, q):
            fn = lambda e: [
                DataLoader._convert_tensor(x) for x in loader.transform_fn(e)
            ]
            futures = [executor.submit(fn, e) for e in batch]
            for fut in as_completed(futures):
                [q.put(x) for x in fut.result()]

        prefetch_batch = 4 * loader.num_workers
        with ThreadPoolExecutor(loader.num_workers) as executor:
            batch = []
            for example in loader.dataset:
                batch.append(example)
                if len(batch) > prefetch_batch:
                    parallel_prefetch(executor, batch, out_queue)
                    batch = []

            if batch:
                parallel_prefetch(executor, batch, out_queue)
        out_queue.join()
        # Signal the main thread that all datasat has been produced.
        done.set()

    def __iter__(self) -> Generator[Dict, None, None]:
        """Use DataLoader as a generator of example."""
        q = Queue(maxsize=64)
        done = threading.Event()

        prefetch_thd = threading.Thread(
            target=DataLoader.prefetch,
            args=(self, q, done),
            daemon=True,
        )
        prefetch_thd.start()

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
        prefetch_thd.join()
        q.join()


def make_video_sampler(sample_gen, video_column_name=None):
    """
    The video sampler makes it easier to directly generate sample frames
    in the DataLoader from a Rikai dataset containing video data.

    For example:
    ```
    data_loader = DataLoader(
        'path/to/video_dataset',
        batch_size=1,
        shuffle=True,
        transform_fn=make_video_sampler(SingleFrameGenerator())
    )
    for row_dict in data_loader:
        print(row_dict['fno']) # frame number
        print(row_dict['frame']) # frame data as np.ndarray
    ```

    Parameters
    ----------
    sample_gen: SampleGenerator
        Generates VideoSampler instances
    video_column_name: str, default None
        Indicate the column name that contains the video. If None then looks
        for the first column with VideoStream or YouTubeVideo data

    Returns
    -------
    fn: Callable[Dict, List(Dict)]
        A function that takes a row data containing a video and returns a list
        of frames sampled from that video
    """
    from rikai.types.video import YouTubeVideo, VideoStream

    def transform_fn(row_dict):
        def process_sample(i, frame):
            frame_dict = row_dict.copy()
            frame_dict["fno"] = i
            frame_dict["frame"] = frame
            return frame_dict

        if video_column_name is not None:
            video = row_dict[video_column_name]
        else:
            for key, value in row_dict.items():
                if isinstance(value, (VideoStream, YouTubeVideo)):
                    video = value
        if isinstance(video, YouTubeVideo):
            video = video.get_stream()
        return [
            DataLoader._convert_tensor(process_sample(i, frame))
            for i, frame in enumerate(sample_gen.get_sampler(video))
        ]

    return transform_fn
