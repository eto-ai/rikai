#  Copyright (c) 2021 Rikai Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from pyspark.sql import Row, SparkSession
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StructField,
    StructType,
)

from rikai.spark.sql.schema import parse_schema


def check_ml_predict(spark: SparkSession, model_name: str):

    # TODO: Replace uri string with Image class after GH#90 is released with
    # the upstream spark
    df = spark.createDataFrame(
        [
            # http://cocodataset.org/#explore?id=484912
            Row(
                uri="http://farm2.staticflickr.com/1129/4726871278_4dd241a03a_z.jpg"  # noqa
            ),
            # https://cocodataset.org/#explore?id=433013
            Row(
                uri="http://farm4.staticflickr.com/3726/9457732891_87c6512b62_z.jpg"  # noqa
            ),
        ],
    )
    df.createOrReplaceTempView("df")

    predictions = spark.sql(
        f"SELECT ML_PREDICT({model_name}, uri) as predictions FROM df"
    )
    predictions.show()
    assert predictions.schema == StructType(
        [
            StructField(
                "predictions",
                StructType(
                    [
                        StructField(
                            "boxes",
                            ArrayType(ArrayType(FloatType())),
                        ),
                        StructField("scores", ArrayType(FloatType())),
                        StructField("labels", ArrayType(IntegerType())),
                    ]
                ),
            ),
        ]
    )
    assert predictions.schema == StructType(
        [
            StructField(
                "predictions",
                parse_schema(
                    "struct<boxes:array<array<float>>, scores:array<float>, labels:array<int>>"  # noqa
                ),
            )
        ]
    )

    assert predictions.count() == 2
