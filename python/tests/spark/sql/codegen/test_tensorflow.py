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
import tensorflow_hub as hub
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
from rikai.contrib.tensorflow.models.ssd import HUB_URL as SSD_HUB_URL
from rikai.contrib.tensorflow.models.ssd import OUTPUT_SCHEMA
from rikai.spark.sql.codegen.fs import FileModelSpec
from rikai.spark.sql.codegen.tensorflow import generate_udf
from rikai.spark.types import Box2dType
from rikai.types import Image


def test_tf_inference_runner(tmp_path: Path):
    m = hub.load(SSD_HUB_URL)
    model_path = str(tmp_path / "model")
    tf.saved_model.save(m, model_path)

    pickler = CloudPickleSerializer()

    spec_path = str(tmp_path / "spec.yml")
    with open(spec_path, "w") as spec_yml:
        spec_yml.write(
            """version: "1.0"
name: ssd
model:
  uri: {}
  flavor: tensorflow
schema: {}
options:
  batch_size: 1
transforms:
  pre: rikai.contrib.tensorflow.models.ssd.pre_processing
  post: rikai.contrib.tensorflow.models.ssd.post_processing
""".format(
                model_path, OUTPUT_SCHEMA
            )
        )

    spec = FileModelSpec(spec_path)
    udf = generate_udf(spec)

    df_iter = [
        pd.Series(
            [
                pickler.dumps(
                    Image(
                        "http://farm2.staticflickr.com/1129/"
                        "4726871278_4dd241a03a_z.jpg"
                    )
                ),
                pickler.dumps(
                    Image(
                        "http://farm4.staticflickr.com/3726/"
                        "9457732891_87c6512b62_z.jpg"
                    )
                ),
            ]
        )
    ]
    output = udf.func(df_iter)

    for batch in output:
        for detections in batch:
            detections = pickler.loads(detections)
            assert len(detections) > 0
            assert set(
                ["detection_boxes", "detection_scores", "detection_classes"]
            ) == set(detections[0].keys())


def test_tf_with_mlflow(tmp_path: Path, spark: SparkSession):
    m = hub.load(SSD_HUB_URL)

    tracking_uri = "sqlite:///" + str(tmp_path / "tracking.db")
    mlflow.set_tracking_uri(tracking_uri)
    spark.conf.set(
        "spark.rikai.sql.ml.registry.mlflow.tracking_uri", tracking_uri
    )

    with mlflow.start_run():
        model_path = str(tmp_path / "model.pt")
        tf.saved_model.save(m, model_path)
        rikai.mlflow.tensorflow.log_model(
            m,
            "model",
            schema=OUTPUT_SCHEMA,
            registered_model_name="tf_ssd",
            pre_processing="rikai.contrib.tensorflow.models"
            + ".ssd.pre_processing",
            post_processing="rikai.contrib.tensorflow.models"
            + ".ssd.post_processing",
        )

    spark.sql("CREATE MODEL ssd USING 'mlflow:///tf_ssd'")
    spark.sql("SHOW MODELS").show()
    df = spark.createDataFrame(
        [
            # http://cocodataset.org/#explore?id=484912
            Row(
                image=Image(
                    "http://farm2.staticflickr.com/1129/"
                    "4726871278_4dd241a03a_z.jpg"
                )
            ),
            # https://cocodataset.org/#explore?id=433013
            Row(
                image=Image(
                    "http://farm4.staticflickr.com/3726/"
                    "9457732891_87c6512b62_z.jpg"
                )
            ),
        ],
    )
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
