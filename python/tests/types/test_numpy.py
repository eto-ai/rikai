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


import numpy as np
from pyspark.ml.linalg import DenseMatrix
from pyspark.sql import SparkSession, Row

# Rikai
from rikai.numpy import wrap
from rikai.types import Box2d, Image


def test_spark_show_numpy(spark: SparkSession, capsys):
    data = wrap(np.random.rand(50, 50, 3))
    data2 = wrap(np.array([1, 2, 3], dtype=np.uint8))
    df = spark.createDataFrame([{"np": data}, {"np": data2}])
    df.show()
    assert np.array_equal(data, df.first().np)
    stdout = capsys.readouterr().out
    print(stdout)
    assert "ndarray(float64" in stdout
    assert "ndarray(uint8" in stdout


def test_readme_example(spark: SparkSession):
    df = spark.createDataFrame(
        [
            {
                "id": 1,
                "mat": DenseMatrix(2, 2, range(4)),
                "image": Image("s3://foo/bar/1.png"),
                "annotations": [
                    Row(
                        label="cat",
                        mask=wrap(np.random.rand(256, 256)),
                        bbox=Box2d(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
                    )
                ],
            }
        ]
    )
    df.show()
