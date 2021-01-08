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

"""A Dataset that supports Rikai UDT convensions
"""

# Standard Library
import importlib
from typing import Any, Dict, List

# Third Party
import pyarrow.parquet as pg
from pyarrow.fs import FileSystem
from pyspark.ml.linalg import Matrix, Vector
from pyspark.sql import Row
from pyspark.sql.types import UserDefinedType

# Rikai
from rikai.logging import logger
from rikai.parquet.resolver import Resolver


class Dataset:
    """Rikai Dataset.

    :py:class:`Dataset` provides read access to a Rikai dataset.

    Parameters
    ----------
    query : str
        A SQL query "SELECT image, annotation FROM s3://foo/bar" or a dataset URI,
        i.e., "s3://foo/bar"
    columns : List[str], optional
        To read only given columns
    shuffle : bool, optional
        Set to true to shuffle the results.
    shuffle_pool_size : int, optional
        The size of the pool for shuffling the examples.
    rank : int
        The rank of this worker in all the distributed workers
    world_size : int
        Total number of distributed workers
    """

    SPARK_PARQUET_ROW_METADATA = b"org.apache.spark.sql.parquet.row.metadata"

    _UDT_CACHE: Dict[str, UserDefinedType] = {}

    def __init__(  # pylint: disable=too-many-arguments
        self,
        query: str,
        columns: List[str] = None,
        shuffle: bool = False,
        shuffle_pool_size: int = 10000,
        rank: int = -1,
        world_size: int = 1,
    ):
        self.uri = query
        self.columns = columns
        self.shuffle = shuffle
        self.shuffle_pool_size = shuffle_pool_size
        self.rank = rank
        self.world_size = world_size

        self.files = Resolver.resolve(self.uri)
        logger.info("Loading parquet files: %s", self.files)

        self.spark_row_metadata = Resolver.get_schema(self.uri)

    @classmethod
    def _find_udt(cls, pyclass: str) -> UserDefinedType:
        """Find UDT class specified by the python class path."""
        if pyclass in cls._UDT_CACHE:
            return cls._UDT_CACHE[pyclass]

        class_path = pyclass.split(".")
        module_name = ".".join(class_path[:-1])
        class_name = class_path[-1]
        try:
            udt_class = getattr(importlib.import_module(module_name), class_name)
        except ImportError as err:
            raise ImportError(f"Could not import user defind type {pyclass}") from err
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
                    # For pyspark.ml.linalg.{Vector,Matrix}, we eagly to convert
                    # them into numpy ndarrays.
                    converted_value = converted_value.toArray()
                converted[name] = converted_value
            elif (
                isinstance(field_type, dict)
                and field_type["type"] == "array"
                and isinstance(field_type["elementType"], dict)
                and field_type["elementType"]["type"] in set(["struct", "udt"])
            ):
                converted[name] = [
                    self._convert(elem, field_type["elementType"])
                    for elem in raw_row[name]
                ]
            else:
                converted[name] = raw_row[name]

        return converted

    def __iter__(self):
        for filepath in self.files:
            fs, path = FileSystem.from_uri(filepath)
            with fs.open_input_file(path) as fobj:
                parquet = pg.ParquetFile(fobj)
                for group_idx in range(parquet.num_row_groups):
                    row_group = parquet.read_row_group(group_idx, columns=self.columns)
                    for batch in row_group.to_batches():  # type: RecordBatch
                        # TODO: read batches not using pandas
                        for _, row in batch.to_pandas().iterrows():
                            yield self._convert(row.to_dict(), self.spark_row_metadata)
