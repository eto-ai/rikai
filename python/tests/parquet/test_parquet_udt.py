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

from pathlib import Path

# Third Party
import numpy as np
import pytest
from pyspark.ml.linalg import DenseMatrix, Vectors
from pyspark.sql import Row
from pyspark.sql.functions import exp
from pyspark.sql.session import SparkSession
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Rikai
from rikai.parquet import Dataset
from rikai.spark.types import Box2dType, NDArrayType
from rikai.testing.asserters import assert_count_equal
from rikai.types import Box2d, Image


def _read_parquets(base_dir):
    return list(Dataset(base_dir))


def test_spark_ml_vectors(spark: SparkSession, tmp_path: Path):
    test_dir = str(tmp_path)
    df = spark.createDataFrame(
        [
            {"name": "a", "vec": Vectors.dense([1, 2])},
            {"name": "b", "vec": Vectors.dense([10])},
        ]
    )
    df.write.mode("overwrite").parquet(str(tmp_path))

    d = spark.read.parquet(test_dir)
    d.show()

    records = _read_parquets(test_dir)
    records = sorted(records, key=lambda x: x["name"])

    expected = [
        {"name": "a", "vec": np.array([1, 2], dtype=np.float64)},
        {"name": "b", "vec": np.array([10], dtype=np.float64)},
    ]

    for exp, rec in zip(expected, records):
        assert exp["name"] == rec["name"]
        assert np.array_equal(exp["vec"], rec["vec"])


def test_spark_ml_matrix(spark: SparkSession, tmp_path: Path):
    test_dir = str(tmp_path)
    df = spark.createDataFrame(
        [
            {"name": 1, "mat": DenseMatrix(2, 2, range(4))},
            {"name": 2, "mat": DenseMatrix(3, 3, range(9))},
        ]
    )
    df.write.mode("overwrite").format("rikai").save(test_dir)
    df.show()

    records = sorted(_read_parquets(test_dir), key=lambda x: x["name"])

    expected = [
        {
            "name": 1,
            "mat": np.array(range(4), dtype=np.float64).reshape(2, 2).T,
        },
        {
            "name": 2,
            "mat": np.array(range(9), dtype=np.float64).reshape(3, 3).T,
        },
    ]
    for exp, rec in zip(expected, records):
        assert exp["name"] == rec["name"]
        assert np.array_equal(exp["mat"], rec["mat"])


def test_images(spark: SparkSession, tmp_path):
    expected = [
        {
            "id": 1,
            "image": Image("s3://123"),
        },
        {
            "id": 2,
            "image": Image("s3://abc"),
        },
    ]
    df = spark.createDataFrame(expected)
    df.write.mode("overwrite").parquet(str(tmp_path))

    records = sorted(_read_parquets(str(tmp_path)), key=lambda x: x["id"])
    assert_count_equal(expected, records)


@pytest.mark.parametrize(
    "data_type",
    [np.int8, np.int64, np.uint16, float, np.float64],
)
def test_numpy(spark: SparkSession, tmp_path, data_type):
    import rikai

    test_dir = str(tmp_path)
    expected = [{"n": rikai.array(range(4), dtype=data_type)}]

    df = spark.createDataFrame(
        expected,
        schema=StructType([StructField("n", NDArrayType(), False)]),
    )
    df.write.mode("overwrite").format("rikai").save(test_dir)

    records = _read_parquets(test_dir)
    assert np.array_equal(np.array(range(4), dtype=data_type), records[0]["n"])


def test_list_of_structs(spark: SparkSession, tmp_path: Path):
    test_dir = str(tmp_path)
    schema = StructType(
        [
            StructField("id", IntegerType(), False),
            StructField(
                "anno",
                ArrayType(
                    StructType(
                        [
                            StructField("label_id", IntegerType(), False),
                            StructField("label", StringType(), False),
                            StructField(
                                "bbox", ArrayType(IntegerType()), False
                            ),
                        ]
                    )
                ),
                False,
            ),
        ]
    )
    df = spark.createDataFrame(
        [
            {
                "id": 1,
                "anno": [
                    {"label": "cat", "label_id": 1, "bbox": [1, 2, 3, 4]},
                    {"label": "dog", "label_id": 2, "bbox": [10, 23]},
                ],
            },
            {
                "id": 2,
                "anno": [
                    {"label": "bug", "label_id": 3, "bbox": [100, 200]},
                    {"label": "aaa", "label_id": 4, "bbox": [-1, -2, -3]},
                ],
            },
        ],
        schema=schema,
    )
    df.repartition(1).write.mode("overwrite").format("rikai").save(test_dir)

    records = _read_parquets(test_dir)
    for expect, actual in zip(
        [
            {
                "id": 1,
                "anno": [
                    {
                        "label": "cat",
                        "label_id": 1,
                        "bbox": np.array([1, 2, 3, 4], dtype=np.int32),
                    },
                    {
                        "label": "dog",
                        "label_id": 2,
                        "bbox": np.array([10, 23], dtype=np.int32),
                    },
                ],
            },
            {
                "id": 2,
                "anno": [
                    {
                        "label": "bug",
                        "label_id": 3,
                        "bbox": np.array([100, 200], dtype=np.int32),
                    },
                    {
                        "label": "aaa",
                        "label_id": 4,
                        "bbox": np.array([-1, -2, -3], dtype=np.int32),
                    },
                ],
            },
        ],
        records,
    ):
        assert expect["id"] == actual["id"]
        assert len(expect["anno"]) == len(actual["anno"])
        assert np.array_equal(
            expect["anno"][0]["bbox"], actual["anno"][0]["bbox"]
        )


def test_bbox(spark: SparkSession, tmp_path: Path):
    test_dir = str(tmp_path)
    df = spark.createDataFrame([Row(b=Box2d(1, 2, 3, 4))])
    df.write.mode("overwrite").format("rikai").save(test_dir)

    records = _read_parquets(test_dir)

    assert_count_equal([{"b": Box2d(1, 2, 3, 4)}], records)


def test_bbox_list(spark: SparkSession, tmp_path: Path):
    test_dir = str(tmp_path)
    df = spark.createDataFrame(
        [Row(bboxes=[Row(b=Box2d(1, 2, 3, 4)), Row(b=Box2d(3, 4, 5, 6))])]
    )
    df.write.mode("overwrite").format("rikai").save(test_dir)

    records = _read_parquets(test_dir)
    assert_count_equal(
        [{"bboxes": [{"b": Box2d(1, 2, 3, 4)}, {"b": Box2d(3, 4, 5, 6)}]}],
        records,
    )


def test_to_pandas(spark: SparkSession, tmp_path: Path):
    test_dir = str(tmp_path)
    spark_df = spark.createDataFrame(
        [Row(bboxes=[Row(b=Box2d(1, 2, 3, 4)), Row(b=Box2d(3, 4, 5, 6))])]
    )
    spark_df.write.mode("overwrite").format("rikai").save(test_dir)
    pandas_df = Dataset(test_dir).to_pandas()
    assert all([isinstance(row["b"], Box2d) for row in pandas_df.bboxes[0]])


def test_struct(spark: SparkSession, tmp_path: Path):
    test_dir = str(tmp_path)
    schema = StructType(
        [
            StructField("id", IntegerType(), False),
            StructField(
                "anno",
                StructType(
                    [
                        StructField("label_id", IntegerType(), False),
                        StructField("label", StringType(), False),
                        StructField("bbox", Box2dType(), False),
                    ]
                ),
                False,
            ),
        ]
    )
    df = spark.createDataFrame(
        [
            {
                "id": 1,
                "anno": {
                    "label": "cat",
                    "label_id": 1,
                    "bbox": Box2d(1, 2, 3, 4),
                },
            },
            {
                "id": 2,
                "anno": {
                    "label": "bug",
                    "label_id": 3,
                    "bbox": Box2d(5, 6, 7, 8),
                },
            },
        ],
        schema=schema,
    )
    df.repartition(1).write.mode("overwrite").format("rikai").save(test_dir)

    pdf = Dataset(test_dir).to_pandas()
    for expect, actual in zip(
        [
            {
                "id": 1,
                "anno": {
                    "label": "cat",
                    "label_id": 1,
                    "bbox": Box2d(1, 2, 3, 4),
                },
            },
            {
                "id": 2,
                "anno": {
                    "label": "bug",
                    "label_id": 3,
                    "bbox": Box2d(5, 6, 7, 8),
                },
            },
        ],
        [row.to_dict() for _, row in pdf.iterrows()],
    ):
        assert expect["id"] == actual["id"]
        assert len(expect["anno"]) == len(actual["anno"])
        assert expect["anno"]["bbox"] == actual["anno"]["bbox"]
