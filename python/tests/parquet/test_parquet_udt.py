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

# Third Party
import numpy as np
from parameterized import parameterized
from pyspark.ml.linalg import DenseMatrix, Vectors
from pyspark.sql import Row
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Rikai
from rikai.parquet import Dataset
from rikai.spark.types import NDArrayType
from rikai.testing import SparkTestCase
from rikai.types import Box2d, Image


class TestParquetUdt(SparkTestCase):
    """Test User Defined Types stored in Parquet"""

    @staticmethod
    def _read_parquets(base_dir):
        return list(Dataset(base_dir))

    def test_spark_ml_vectors(self):
        df = self.spark.createDataFrame(
            [
                {"name": "a", "vec": Vectors.dense([1, 2])},
                {"name": "b", "vec": Vectors.dense([10])},
            ]
        )
        df.write.mode("overwrite").parquet(self.test_dir)

        d = self.spark.read.parquet(self.test_dir)
        d.show()

        records = self._read_parquets(self.test_dir)
        records = sorted(records, key=lambda x: x["name"])

        expected = [
            {"name": "a", "vec": np.array([1, 2], dtype=np.float64)},
            {"name": "b", "vec": np.array([10], dtype=np.float64)},
        ]

        for exp, rec in zip(expected, records):
            self.assertEqual(exp["name"], rec["name"])
            self.assertTrue(
                np.array_equal(exp["vec"], rec["vec"]),
                f"Expected {exp['vec']}({exp['vec'].dtype}) Got {rec['vec']}({rec['vec'].dtype})",
            )

    def test_spark_ml_matrix(self):
        df = self.spark.createDataFrame(
            [
                {"name": 1, "mat": DenseMatrix(2, 2, range(4))},
                {"name": 2, "mat": DenseMatrix(3, 3, range(9))},
            ]
        )
        df.write.mode("overwrite").format("rikai").save(self.test_dir)
        df.show()

        records = sorted(self._read_parquets(self.test_dir), key=lambda x: x["name"])

        expected = [
            {"name": 1, "mat": np.array(range(4), dtype=np.float64).reshape(2, 2).T},
            {"name": 2, "mat": np.array(range(9), dtype=np.float64).reshape(3, 3).T},
        ]
        for exp, rec in zip(expected, records):
            self.assertEqual(exp["name"], rec["name"])
            self.assertTrue(np.array_equal(exp["mat"], rec["mat"]))

    def test_images(self):
        expected = [
            {
                "id": 1,
                "image": Image(uri="s3://123"),
            },
            {
                "id": 2,
                "image": Image(uri="s3://abc"),
            },
        ]
        df = self.spark.createDataFrame(expected)
        df.write.mode("overwrite").parquet(self.test_dir)

        records = sorted(self._read_parquets(self.test_dir), key=lambda x: x["id"])
        self.assertCountEqual(expected, records)

    @parameterized.expand(
        [(np.int8,), (np.int64,), (np.uint16,), (float,), (np.float64,)]
    )
    def test_numpy(self, data_type):
        import rikai

        expected = [{"n": rikai.array(range(4), dtype=data_type)}]

        df = self.spark.createDataFrame(
            expected, schema=StructType([StructField("n", NDArrayType(), False)])
        )
        df.write.mode("overwrite").format("rikai").save(self.test_dir)

        records = self._read_parquets(self.test_dir)
        self.assertTrue(
            np.array_equal(np.array(range(4), dtype=data_type), records[0]["n"])
        )

    def test_list_of_structs(self):
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
                                StructField("bbox", ArrayType(IntegerType()), False),
                            ]
                        )
                    ),
                    False,
                ),
            ]
        )
        df = self.spark.createDataFrame(
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
        df.repartition(1).write.mode("overwrite").format("rikai").save(self.test_dir)

        records = self._read_parquets(self.test_dir)
        self.assertTrue(
            [
                {
                    "id": 1,
                    "anno": [
                        {
                            "label": "cat",
                            "label_id": 1,
                            "bbox": np.array([1, 2, 3, 4]),
                        },
                        {"label": "dog", "label_id": 2, "bbox": np.array([10, 23])},
                    ],
                },
                {
                    "id": 2,
                    "anno": [
                        {
                            "label": "bug",
                            "label_id": 3,
                            "bbox": np.array([100, 200]),
                        },
                        {
                            "label": "aaa",
                            "label_id": 4,
                            "bbox": np.array([-1, -2, -3]),
                        },
                    ],
                },
            ],
            records,
        )

    def test_bbox(self):
        df = self.spark.createDataFrame([Row(b=Box2d(1, 2, 3, 4))])
        df.write.mode("overwrite").format("rikai").save(self.test_dir)

        records = self._read_parquets(self.test_dir)

        self.assertCountEqual([{"b": Box2d(1, 2, 3, 4)}], records)

    def test_bbox_list(self):
        df = self.spark.createDataFrame(
            [Row(bboxes=[Row(b=Box2d(1, 2, 3, 4)), Row(b=Box2d(3, 4, 5, 6))])]
        )
        df.write.mode("overwrite").format("rikai").save(self.test_dir)

        records = self._read_parquets(self.test_dir)
        self.assertCountEqual(
            [{"bboxes": [{"b": Box2d(1, 2, 3, 4)}, {"b": Box2d(3, 4, 5, 6)}]}], records
        )
