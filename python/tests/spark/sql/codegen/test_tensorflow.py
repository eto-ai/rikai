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
import tensorflow as tf
import tensorflow_hub as hub
from pyspark.sql import Row, SparkSession
from tensorflow import keras

import rikai
from rikai.types import Image


def test_tf_ssd_model(tmp_path: Path, spark: SparkSession):
    HUB_URL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    m = hub.load(HUB_URL)

    tracking_uri = "sqlite:///" + str(tmp_path / "tracking.db")
    mlflow.set_tracking_uri(tracking_uri)
    spark.conf.set(
        "spark.rikai.sql.ml.registry.mlflow.tracking_uri", tracking_uri
    )

    with mlflow.start_run():
        model_path = str(tmp_path / "model.pt")
        tf.saved_model.save(m, model_path)
        # m.save(model_path)
        rikai.mlflow.tensorflow.log_model(
            m,
            "model",
            schema="array<float>",
            registered_model_name="tfssd",
        )

    spark.sql("CREATE MODEL ssd USING 'mlflow:///tfssd'")
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

    spark.sql("SELECT ML_PREDICT(ssd, image) FROM images").show()

    # print(
    #     m(
    #         [
    #             Image(
    #                 "http://farm4.staticflickr.com/3726/"
    #                 "9457732891_87c6512b62_z.jpg"
    #             ).to_numpy(),
    #         ]
    #     )
    # )
