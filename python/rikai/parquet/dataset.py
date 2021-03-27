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

"""Parquet-based Rikai Dataset that supports automatically UDT conversions.
"""

# Standard Library
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third Party
import pyarrow.parquet as pg
from pyarrow.fs import FileSystem
from pyspark.ml.linalg import Matrix, Vector
from pyspark.sql import Row
from pyspark.sql.types import UserDefinedType

# Rikai
from rikai.io import open_input_stream
from rikai.logging import logger
from rikai.parquet.resolver import Resolver
from rikai.parquet.shuffler import RandomShuffler

__all__ = ["Dataset"]


class Dataset:
    """Rikai Dataset.

    :py:class:`Dataset` provides read access to a Rikai dataset. It supports

    - Read Rikai encoded dataset on the supported storage medias, i.e.,
      local filesystem, AWS S3 or Google GCS.
    - Automatically deserialize data into semantic user defined types (UDT).
    - Shuffle the dataset randomly through ``shuffle`` flag.
    - Distributed training by setting ``world_size`` and ``rank`` parameters.
      When enabled, parquet `row-group` level partition will be used to
      distribute data amount the workers.

    Parameters
    ----------
    query : str
        A SQL query "SELECT image, annotation FROM s3://foo/bar" or
        a dataset URI, i.e., "s3://foo/bar"
    columns : List[str], optional
        To read only given columns
    shuffle : bool, optional
        Set to True to shuffle the results.
    shuffler_capacity : int
        The size of the buffer to shuffle the examples. The size of buffer
        does not impact the distribution of possibility that an example
        is picked.
    seed : int, optional
        Random seed for shuffling process.
    world_size : int
        Total number of distributed workers
    rank : int
        The rank of this worker in all the distributed workers

    Notes
    -----
    - Typically user should not directly use this class. Instead users are
      encouraged to use framework-native readers, for example, using
      :py:class:`rikai.torch.data.DataLoader` in
      `Pytorch <https://pytorch.org/>`_
    """

    SPARK_PARQUET_ROW_METADATA = b"org.apache.spark.sql.parquet.row.metadata"

    _UDT_CACHE: Dict[str, UserDefinedType] = {}

    def __init__(  # pylint: disable=too-many-arguments
        self,
        query: Union[str, Path],
        columns: Optional[List[str]] = None,
        shuffle: bool = False,
        shuffler_capacity: int = 128,
        seed: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
    ):
        self.uri = str(query)
        self.columns = columns
        self.shuffle = shuffle
        self.shuffler_capacity = shuffler_capacity
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        if self.world_size > 1:
            logger.info(
                "Running in distributed mode, world size=%s, rank=%s",
                world_size,
                rank,
            )

        # Provide deterministic order between distributed workers.
        self.files = sorted(Resolver.resolve(self.uri))
        if self.rank == 0:
            logger.info("Loading parquet files: %s", self.files)

        self.spark_row_metadata = Resolver.get_schema(self.uri)

    def __repr__(self) -> str:
        return "Dataset(uri={}, columns={}, shuffle={})".format(
            self.uri, self.columns if self.columns else "[*]", self.shuffle
        )

    @classmethod
    def _find_udt(cls, pyclass: str) -> UserDefinedType:
        """Find UDT class specified by the python class path."""
        if pyclass in cls._UDT_CACHE:
            return cls._UDT_CACHE[pyclass]

        class_path = pyclass.split(".")
        module_name = ".".join(class_path[:-1])
        class_name = class_path[-1]
        try:
            udt_class = getattr(
                importlib.import_module(module_name), class_name
            )
        except ImportError as err:
            raise ImportError(
                f"Could not import user defind type {pyclass}"
            ) from err
        cls._UDT_CACHE[pyclass] = udt_class()
        return cls._UDT_CACHE[pyclass]

    def _convert(
        self, raw_row: Dict[str, Any], schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert Spark UDT to native rikia or numpy types.

        Parameters
        ----------
        raw_row : Dict[str, Any]
        schema : Dict[str, Any]
            Spark schema in the JSON format
        """
        assert schema["type"] == "struct"
        converted = {}
        for field in schema["fields"]:
            name = field["name"]
            if name not in raw_row:
                # This column is not selected, skip
                continue
            field_type = field["type"]
            if isinstance(field_type, dict) and field_type["type"] == "udt":
                udt = self._find_udt(field_type["pyClass"])

                value = raw_row[name]
                if isinstance(value, dict):
                    row = Row(**value)
                else:
                    row = Row(value)
                converted_value = udt.deserialize(row)
                if isinstance(converted_value, (Vector, Matrix)):
                    # For pyspark.ml.linalg.{Vector,Matrix}, we eagly convert
                    # them into numpy ndarrays.
                    converted_value = converted_value.toArray()
                converted[name] = converted_value
            elif (
                isinstance(field_type, dict)
                and field_type["type"] == "array"
                and isinstance(field_type["elementType"], dict)
                and field_type["elementType"]["type"] in {"struct", "udt"}
            ):
                converted[name] = [
                    self._convert(elem, field_type["elementType"])
                    for elem in raw_row[name]
                ]
            else:
                converted[name] = raw_row[name]

        return converted

    def __iter__(self):
        shuffler = RandomShuffler(
            self.shuffler_capacity if self.shuffle else 1, self.seed
        )
        group_count = -1
        for file_uri in self.files:
            with open_input_stream(file_uri) as fobj:
                parquet = pg.ParquetFile(fobj)
                for group_idx in range(parquet.num_row_groups):
                    # A simple form of row-group level bucketing without
                    # memory overhead.
                    # Pros:
                    #  - It requires zero communication to initialize the
                    #    distributed policy
                    #  - It uses little memory and no startup overhead, i.e.
                    #    collecting row groups.
                    # Cons:
                    #   The drawback would be if the world size is much larger
                    #   than the average number of row groups. As a result,
                    #   many of the file open operations would be wasted.
                    group_count += 1
                    if group_count % self.world_size != self.rank:
                        continue
                    row_group = parquet.read_row_group(
                        group_idx, columns=self.columns
                    )
                    for (
                        batch
                    ) in row_group.to_batches():  # type: pyarrow.RecordBatch
                        # TODO: read batches not using pandas
                        for _, row in batch.to_pandas().iterrows():
                            shuffler.append(row)
                            # Maintain the shuffler buffer around its capacity.

                            while shuffler.full():
                                yield self._convert(
                                    shuffler.pop().to_dict(),
                                    self.spark_row_metadata,
                                )
        while shuffler:
            yield self._convert(
                shuffler.pop().to_dict(), self.spark_row_metadata
            )
