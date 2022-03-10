#  Copyright (c) 2022 Rikai Authors
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


from pathlib import Path

import mlflow
import pandas as pd
import tensorflow as tf
from pyspark.serializers import CloudPickleSerializer
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StructField,
    StructType,
)

import rikai
from rikai.spark.sql.codegen.fs import FileModelSpec
from rikai.spark.sql.codegen.mlflow_registry import CONF_MLFLOW_TRACKING_URI
from rikai.spark.types import Box2dType
from rikai.types import Image
from rikai.testing.utils import apply_model_spec


def test_tf_inference_runner(
    tfhub_ssd, tmp_path: Path, two_flickr_images: list
):
    m, model_path = tfhub_ssd
    spec_path = str(tmp_path / "spec.yml")
    with open(spec_path, "w") as spec_yml:
        spec_yml.write(
            f"""version: "1.0"
name: ssd
model:
  uri: {model_path}
  flavor: tensorflow
  type: rikai.contrib.tfhub.tensorflow.ssd
options:
  batch_size: 1
"""
        )

    spec = FileModelSpec(spec_path)
    inputs = [pd.Series(image) for image in two_flickr_images]
    results = apply_model_spec(spec, inputs)

    for series in results:
        assert len(series) == 1
        assert len(series[0]) == 100


def test_tf_with_mlflow(
    tfhub_ssd, tmp_path: Path, spark: SparkSession, two_flickr_rows: list
):
    m, model_path = tfhub_ssd

    tracking_uri = "sqlite:///" + str(tmp_path / "tracking.db")
    mlflow.set_tracking_uri(tracking_uri)
    spark.conf.set(CONF_MLFLOW_TRACKING_URI, tracking_uri)

    with mlflow.start_run():
        model_path = str(tmp_path / "model.pt")
        tf.saved_model.save(m, model_path)
        rikai.mlflow.tensorflow.log_model(
            m,
            "model",
            model_type="rikai.contrib.tfhub.tensorflow.ssd",
            registered_model_name="tf_ssd",
        )

    spark.sql("CREATE MODEL ssd USING 'mlflow:///tf_ssd'")
    spark.sql("SHOW MODELS").show()
    df = spark.createDataFrame(two_flickr_rows)
    df.createOrReplaceTempView("images")

    df = spark.sql("SELECT ML_PREDICT(ssd, image) as det FROM images")
    df.show()
    df.createOrReplaceTempView("detections")
    assert df.schema == StructType(
        [
            StructField(
                "det",
                ArrayType(
                    StructType(
                        [
                            StructField("detection_boxes", Box2dType()),
                            StructField("detection_scores", FloatType()),
                            StructField("detection_classes", IntegerType()),
                        ]
                    )
                ),
            )
        ]
    )
    assert (
        spark.sql("SELECT det FROM detections WHERE size(det) > 1").count()
        == 2
    )
