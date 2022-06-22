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
import random

import numpy as np
import pandas as pd
from pyarrow.lib import ArrowInvalid
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pytest

from rikai.parquet.dataset import Dataset
from rikai.parquet.writer import df_to_rikai
from rikai.spark.types import *
from rikai.types import *

IMG_ARR = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)


def test_roundtrip(spark: SparkSession, tmp_path: Path):
    df, schema = _make_df()
    df_to_rikai(df, str(tmp_path), schema)

    pandas_df = pd.DataFrame(Dataset(str(tmp_path)))
    assert isinstance(pandas_df.image[0], Image)
    assert isinstance(pandas_df.embedded_image[0], Image)
    assert (pandas_df.embedded_image[0].to_numpy() == IMG_ARR).all()
    spark_df = spark.read.format("rikai").load(str(tmp_path))
    assert schema.json() == spark_df.schema.json()


def test_partition_cols(tmp_path: Path):
    df, schema = _make_df(1000)
    df_to_rikai(
        df, str(tmp_path), schema, partition_cols="split", max_rows_per_file=10
    )
    _verify_file_size(Path(tmp_path), max_rows_per_file=10)
    partition_info = [(k, df[k].unique()) for k in ["split"]]
    _verify_partitioning(Path(tmp_path), partition_info)


def test_mode(spark: SparkSession, tmp_path: Path):
    df, schema = _make_df(1000)
    df_to_rikai(
        df[df.split == "train"],
        str(tmp_path),
        schema,
        partition_cols="split",
        max_rows_per_file=10,
    )
    df1 = pd.DataFrame(Dataset(str(tmp_path)))
    assert len(df1) == len(df[df.split == "train"])

    # default is 'error'
    with pytest.raises(ArrowInvalid):
        df_to_rikai(
            df,
            str(tmp_path),
            schema,
            partition_cols="split",
            max_rows_per_file=10,
        )

    # overwrite_or_ignore
    df_to_rikai(
        df,
        str(tmp_path),
        schema,
        partition_cols="split",
        max_rows_per_file=10,
        mode="overwrite_or_ignore",
    )
    df2 = pd.DataFrame(Dataset(str(tmp_path)))
    assert len(df2) == len(df)

    # delete_matching should replace
    df["image_id"] = df.image_id.apply(lambda x: str(-int(x)))
    df_to_rikai(
        df,
        str(tmp_path),
        schema,
        partition_cols="split",
        max_rows_per_file=10,
        mode="delete_matching",
    )
    pdf = pd.DataFrame(Dataset(str(tmp_path)))
    assert pdf.image_id.apply(int).apply(lambda x: x <= 0).all()


def _make_df(nrows=1):
    schema = StructType(
        fields=[
            StructField("image_id", StringType()),
            StructField("image", ImageType()),
            StructField("embedded_image", ImageType()),
            StructField("image_labels", ArrayType(elementType=StringType())),
            StructField("split", StringType()),
            StructField(
                "annotations",
                ArrayType(
                    elementType=StructType(
                        fields=[
                            StructField("label", StringType()),
                            StructField("box", Box2dType()),
                        ]
                    )
                ),
            ),
        ]
    )

    labels = ["car", "truck", "person", "traffic light"]
    splits = ["train", "val", "test"]
    df = pd.DataFrame(
        [
            {
                "image_id": str(i),
                "image": Image(f"s3://bucket/path_{i}.jpg"),
                "embedded_image": Image.from_array(IMG_ARR),
                "image_labels": ["foo", "bar"],
                "split": splits[random.randint(0, len(splits) - 1)],
                "annotations": [
                    {
                        "label": labels[random.randint(0, len(labels) - 1)],
                        "box": Box2d(
                            xmin=0,
                            ymin=0,
                            xmax=random.random() * 100 + 1,
                            ymax=random.random() * 100 + 1,
                        ),
                    }
                ],
            }
            for i in np.arange(nrows)
        ]
    )
    return df, schema


def _verify_file_size(dest: Path, max_rows_per_file: int):
    parquet_files = list(dest.glob("*.parquet"))
    for filepath in parquet_files:
        m = pq.ParquetFile(filepath).metadata
        assert m.num_rows <= max_rows_per_file


def _verify_partitioning(dest: Path, partition_info: list):
    for part_dir in dest.iterdir():
        col, value = part_dir.name.split("=")
        assert col == partition_info[0][0]
        assert value in partition_info[0][1]
        if len(partition_info) > 1:
            _verify_partitioning(part_dir, partition_info[1:])
