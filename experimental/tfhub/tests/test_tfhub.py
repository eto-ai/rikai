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

import pandas as pd
from pyspark.sql import SparkSession

from rikai.spark.functions import to_image
from rikai.testing.utils import apply_model_spec
from rikai.types.vision import Image

work_dir = Path().absolute().parent.parent
image_path = f"{work_dir}/python/tests/assets/test_image.jpg"


def test_ssd_model_type():
    inputs_list = [pd.Series(
        [Image(image_path), Image(image_path), Image(image_path)])]
    # inputs_list = [pd.Series(Image(image_path))]
    results_list = apply_model_spec(
        {
            "name": "tfssd",
            "uri": f"tfhub:///tensorflow/ssd_mobilenet_v2/2",
            "modelType": "ssd",
        },
        inputs_list,
    )
    assert len(results_list) == 1
    series = results_list[0]
    assert len(series[0]) == 100


def test_ssd(spark: SparkSession):
    spark.udf.register("to_image", to_image)
    spark.sql(
        f"""
        CREATE MODEL tfssd
        MODEL_TYPE ssd
        OPTIONS (device="cpu", batch_size=32)
        USING "tfhub:///tensorflow/ssd_mobilenet_v2/2";
        """
    )
    result = spark.sql(
        f"""
    select ML_PREDICT(tfssd, to_image('{image_path}')) as preds
    """
    )
    assert result.count() == 1


def test_multi_pics_ssd(spark: SparkSession):
    spark.udf.register("to_image", to_image)
    spark.sql(
        f"""
        CREATE MODEL tfssd
        MODEL_TYPE ssd
        OPTIONS (device="cpu", batch_size=32)
        USING "tfhub:///tensorflow/ssd_mobilenet_v2/2";
        """
    )

    spark.range(10).selectExpr(
        "id as id", f"to_image('{image_path}') as image"
    ).createOrReplaceTempView("test_view")

    result = spark.sql(
        f"""
    select id, ML_PREDICT(tfssd, image) as preds from test_view
    """
    )
    result.show()
    assert result.count() == 10
