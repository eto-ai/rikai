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

"""Parquet-based Rikai Dataset that supports automatically UDT conversions.
"""

# Standard Library
import importlib
import json
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

# Third Party
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow import fs
from pyspark.ml.linalg import Matrix, Vector
from pyspark.sql import Row
from pyspark.sql.types import UserDefinedType

# Rikai
from rikai.exceptions import ColumnNotFoundError
from rikai.io import exists, open_input_stream, open_uri
from rikai.logging import logger
from rikai.mixin import ToNumpy, ToPIL
from rikai.parquet.resolver import Resolver

__all__ = ["Dataset"]


class Dataset:
    """Rikai Dataset.

    :py:class:`Dataset` provides read access to a Rikai dataset. It supports

    - Read Rikai encoded dataset on the supported storage medias, i.e.,
      local filesystem, AWS S3 or Google GCS.
    - Automatically deserialize data into semantic user defined types (UDT).
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
    seed : int, optional
        Random seed for shuffling process.
    world_size : int
        Total number of distributed workers
    rank : int
        The rank of this worker in all the distributed workers
    offset : int
        Instruct the dataset to skip the first N records.

    Notes
    -----
    - Typically user should not directly use this class. Instead, users are
      encouraged to use framework-native readers, for example, using
      :py:class:`rikai.pytorch.data.Dataset` in
      `Pytorch <https://pytorch.org/>`_
    """

    SPARK_PARQUET_ROW_METADATA = b"org.apache.spark.sql.parquet.row.metadata"

    _UDT_CACHE: Dict[str, UserDefinedType] = {}

    def __init__(  # pylint: disable=too-many-arguments
        self,
        query: Union[str, Path],
        columns: Optional[List[str]] = None,
        seed: Optional[int] = None,
        world_size: int = 1,
        rank: int = 0,
        offset: int = 0,
    ):
        self.uri = str(query)
        self.columns = columns
        self.seed = seed
        self.rank = rank
        self.offset = offset
        if offset < 0:
            raise ValueError("Offset must be non-negative value")

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

        if columns:
            # TODO: check nested columns
            for col in columns:
                self._check_column(col, self.spark_row_metadata)

    def __repr__(self) -> str:
        return "Dataset(uri={}, columns={})".format(
            self.uri, self.columns if self.columns else "[*]"
        )

    @staticmethod
    def _check_column(column: str, schema: dict):
        for field in schema["fields"]:
            if field["name"] == column:
                break
        else:
            raise ColumnNotFoundError(f"Column not found: {column}")

    @property
    def metadata(self) -> dict:
        """Rikai metadata"""
        metadata_path = os.path.join(self.uri, "_rikai", "metadata.json")

        if not exists(metadata_path):
            return {}
        with open_uri(metadata_path) as fobj:
            return json.load(fobj)

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
        """Convert Spark UDT to native rikai or numpy types.

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
            if not isinstance(field_type, dict):
                converted[name] = raw_row[name]
            elif field_type["type"] == "udt":
                udt = self._find_udt(field_type["pyClass"])
                converted[name] = _convert_udt_value(raw_row[name], udt)
            elif field_type["type"] == "array" and isinstance(
                field_type["elementType"], dict
            ):
                converted[name] = self._convert_array(
                    raw_row[name], field_type["elementType"]
                )
            elif field_type["type"] == "struct":
                converted[name] = {
                    f["name"]: self._convert(raw_row[f["name"]], f["type"])
                    for f in field_type["type"]["fields"]
                }
            else:
                converted[name] = raw_row[name]

        return converted

    def __iter__(self):
        offset = self.offset
        group_count = -1
        for file_uri in self.files:
            with open_input_stream(file_uri) as fobj:
                parquet = pq.ParquetFile(fobj)
                file_metadata: pq.FileMetaData = parquet.metadata
                if offset >= file_metadata.num_rows:
                    # Skipping files.
                    offset -= parquet.metadata.num_rows
                    continue
                for group_idx in range(parquet.num_row_groups):
                    row_metadata: pq.RowGroupMetaData = (
                        file_metadata.row_group(group_idx)
                    )
                    if offset >= row_metadata.num_rows:
                        # Skip row groups
                        offset -= row_metadata.num_rows
                        continue

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
                        for row in batch.to_pylist()[offset:]:
                            yield self._convert(
                                row,
                                self.spark_row_metadata,
                            )

    def to_pandas(self, limit=None):
        """Create a pandas dataframe from the parquet data in this Dataset

        Parameters
        ----------
        limit: int, default None
            The max number of rows to retrieve. If none, 0, or negative
            then retrieve all rows
        """
        filesystem, path = fs.FileSystem.from_uri(self.uri)
        dataset = ds.dataset(path, filesystem=filesystem, format="parquet")
        if limit is None or limit <= 0:
            raw_df = dataset.to_table(columns=self.columns).to_pandas()
        else:
            raw_df = dataset.head(limit, columns=self.columns).to_pandas()
        types = {
            f["name"]: f["type"] for f in self.spark_row_metadata["fields"]
        }
        return pd.DataFrame(
            {
                name: self._convert_col(col, types.get(name))
                for name, col in raw_df.iteritems()
            }
        )

    def _convert_col(self, col: pd.Series, field_type) -> pd.Series:
        if field_type is None:
            return col
        if not isinstance(field_type, dict):
            return col
        elif field_type["type"] == "udt":
            udt = self._find_udt(field_type["pyClass"])
            return col.apply(partial(_convert_udt_value, udt=udt))
        elif field_type["type"] == "struct":
            return col.apply(lambda d: self._convert(d, field_type))
        elif field_type["type"] == "array":
            return col.apply(
                lambda d: self._convert_array(d, field_type["elementType"])
            )
        else:
            return col

    def _convert_array(self, arr, element_type):
        return [self._convert(x, element_type) for x in arr]


def _convert_udt_value(value, udt):
    if isinstance(value, dict):
        row = Row(**value)
    else:
        row = Row(value)
    converted_value = udt.deserialize(row)
    if isinstance(converted_value, (Vector, Matrix)):
        converted_value = converted_value.toArray()
    return converted_value


def convert_tensor(row, use_pil: bool = False):
    """
    Convert a parquet row into tensors.

    If use_pil is set to True, this method returns a PIL image instead,
    and relies on the customer code to convert PIL image to tensors.
    """
    if use_pil and isinstance(row, ToPIL):
        return row.to_pil()
    elif isinstance(row, ToNumpy):
        return row.to_numpy()
    elif not isinstance(row, (Mapping, pd.Series)):
        # Primitive values
        return row

    tensors = {}
    for key, value in row.items():
        if isinstance(value, dict):
            tensors[key] = convert_tensor(value)
        elif isinstance(value, (list, tuple)):
            tensors[key] = np.array([convert_tensor(elem) for elem in value])
        elif use_pil and isinstance(value, ToPIL):
            tensors[key] = value.to_pil()
        elif isinstance(value, ToNumpy):
            tensors[key] = value.to_numpy()
        else:
            tensors[key] = value
    return tensors
