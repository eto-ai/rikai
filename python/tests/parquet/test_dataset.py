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

from pathlib import Path

# Third Party
import numpy as np
import pyarrow.parquet as pq
import pytest
from pyspark.sql import Row, SparkSession

# Rikai
from rikai.parquet import Dataset
from rikai.testing.asserters import assert_count_equal
from rikai.types import Image


def _select_columns(spark: SparkSession, tmpdir: str):
    df = spark.createDataFrame(
        [
            Row(id=1, col1="value", col2=123),
            Row(id=2, col1="more", col2=456),
        ]
    )
    if tmpdir.startswith("s3://"):
        df.write.format("rikai").save("s3a" + tmpdir[2:])
    else:
        df.write.format("rikai").save(tmpdir)

    dataset = Dataset(tmpdir, columns=["id", "col1"])
    actual = sorted(list(dataset), key=lambda x: x["id"])

    assert_count_equal(
        [{"id": 1, "col1": "value"}, {"id": 2, "col1": "more"}], actual
    )


def test_select_columns(spark: SparkSession, tmp_path: Path):
    """Test reading rikai dataset with selected columns."""
    _select_columns(spark, str(tmp_path))


def test_offset(spark: SparkSession, tmp_path: Path):
    dest = str(tmp_path)
    df = spark.createDataFrame(
        [Row(id=i, col=f"val-{i}") for i in range(1000)]
    )
    df.write.format("rikai").save(dest)

    data1 = Dataset(dest)
    row = next(iter(data1))
    assert row["id"] == 0

    data2 = Dataset(dest, offset=10)
    row = next(iter(data2))
    assert row["id"] == 10

    with pytest.raises(StopIteration):
        next(iter(Dataset(dest, offset=2000)))  # off the edge


def _verify_group_size(dest: Path, group_size: int):
    parquet_files = list(dest.glob("*.parquet"))
    total_size = 0
    for filepath in parquet_files:
        pqfile = pq.ParquetFile(filepath)
        file_metadata: pq.FileMetaData = pqfile.metadata
        for row_id in range(file_metadata.num_row_groups):
            row_group: pq.RowGroupMetaData = file_metadata.row_group(row_id)
            assert row_group.total_byte_size <= group_size
            total_size += row_group.total_byte_size
    assert total_size >= group_size * len(parquet_files)


def test_group_size(spark: SparkSession, tmp_path: Path):
    dest = str(tmp_path)
    df = spark.createDataFrame(
        [
            Row(
                id=i,
                label=f"label-{i}",
                image=Image.from_array(
                    np.random.randint(
                        0, 128, size=(64, 64, 3), dtype=np.uint8
                    ),
                ),
            )
            for i in range(10000)
        ]
    )
    df.write.format("rikai").save(dest)
    _verify_group_size(tmp_path, 32 * 1024 * 1024)  # Default group size

    (
        df.write.format("rikai")
        .option("rikai.block.size", 8 * 1024 * 1024)
        .mode("overwrite")
        .save(dest)
    )
    _verify_group_size(tmp_path, 8 * 1024 * 1024)


def test_select_columns_on_gcs(spark: SparkSession, gcs_tmpdir: str):
    _select_columns(spark, gcs_tmpdir)


def test_select_over_s3(spark: SparkSession, s3_tmpdir: str):
    _select_columns(spark, s3_tmpdir)
