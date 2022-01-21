#  Copyright 2021 Rikai Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from pathlib import Path

import numpy as np
from pyspark.ml.linalg import DenseMatrix
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import udf
import PIL

# Rikai
from rikai.numpy import view
from rikai.spark.types import NDArrayType
from rikai.types import Box2d, Image


def test_spark_show_numpy(spark: SparkSession, capsys):
    data = view(np.random.rand(50, 50, 3))
    data2 = view(np.array([1, 2, 3], dtype=np.uint8))
    df = spark.createDataFrame([{"np": data}, {"np": data2}])
    df.show()
    assert np.array_equal(data, df.first().np)
    stdout = capsys.readouterr().out
    print(stdout)
    assert "ndarray(float64" in stdout
    assert "ndarray(uint8" in stdout


def test_readme_example(spark: SparkSession, tmp_path: Path):
    df = spark.createDataFrame(
        [
            {
                "id": 1,
                "mat": DenseMatrix(2, 2, range(4)),
                "image": Image("s3://foo/bar/1.png"),
                "annotations": [
                    Row(
                        label="cat",
                        mask=view(np.random.rand(256, 256)),
                        bbox=Box2d(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
                    )
                ],
            }
        ]
    )
    df.write.format("rikai").save(str(tmp_path))


def test_numpy_in_udf(spark: SparkSession, tmp_path: Path):
    df = spark.createDataFrame([Row(data=view(np.random.rand(256, 256)))])

    @udf(returnType=NDArrayType())
    def resize_mask(arr: np.ndarray) -> np.ndarray:
        img = PIL.Image.fromarray(arr)
        resized_img = img.resize((32, 32))
        return view(np.asarray(resized_img))

    df = df.withColumn("resized", resize_mask("data"))
    df.show()

    df.write.format("rikai").save(str(tmp_path))
