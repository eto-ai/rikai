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

import os
import urllib
from pathlib import Path

import mlflow
import torch
import yolov5
from pyspark.sql import Row, SparkSession

import rikai
from rikai.contrib.yolov5.transforms import OUTPUT_SCHEMA
from rikai.types import Image


def test_yolov5(tmp_path: Path, spark: SparkSession):
    tmp_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = "sqlite:///" + str(tmp_path / "tracking.db")
    mlflow.set_tracking_uri(tracking_uri)
    registered_model_name = f"yolov5s-model"
    mlflow.set_experiment(registered_model_name)

    with mlflow.start_run():
        ####
        # Part 1: Train the model and register it on MLflow
        ####
        pretrained = str(tmp_path / "yolov5s.pt")
        url = (
            "https://github.com/ultralytics/yolov5/"
            + "releases/download/v5.0/yolov5s.pt"
        )
        if os.path.exists(pretrained):
            urllib.request.urlretrieve(url, pretrained)
        model = yolov5.load(pretrained)
        pre = "rikai.contrib.yolov5.transforms.pre_processing"
        post = "rikai.contrib.yolov5.transforms.post_processing"
        if torch.cuda.is_available():
            print("Using GPU\n")
            device = "gpu"
        else:
            print("Using CPU\n")
            device = "cpu"

        spark.conf.set(
            "rikai.sql.ml.registry.mlflow.tracking_uri", tracking_uri
        )
        work_dir = Path().absolute().parent.parent
        image_path = f"{work_dir}/python/tests/assets/test_image.jpg"

        # Use case 1: log model with customized flavor
        rikai.mlflow.pytorch.log_model(
            model,
            "model",
            OUTPUT_SCHEMA,
            pre_processing=pre,
            post_processing=post,
            registered_model_name="yolov5_way_1",
            customized_flavor="yolov5",
        )
        spark.sql(
            f"""
        CREATE MODEL yolov5_way_1
        OPTIONS (
            device='{device}',
            iou_thres=0.5
        )
        USING 'mlflow:///yolov5_way_1'
        """
        )
        spark.createDataFrame(
            [Row(image=Image(image_path))]
        ).createOrReplaceTempView("images")
        result1 = spark.sql(
            f"""
        select ML_PREDICT(yolov5_way_1, image) as pred FROM images
        """
        )

        ####
        # Part 2: create the model using the registered MLflow uri
        ####
        rikai.mlflow.pytorch.log_model(
            model,
            "model",
            OUTPUT_SCHEMA,
            pre_processing=pre,
            post_processing=post,
            registered_model_name="yolov5_way_2",
        )
        spark.sql(
            f"""
        CREATE MODEL yolov5_way_2
        FLAVOR yolov5
        OPTIONS (
            device='{device}'
        )
        USING 'mlflow:///yolov5_way_2'
        """
        )
        result2 = spark.sql(
            f"""
        select ML_PREDICT(yolov5_way_2, image) as pred FROM images
        """
        )

        assert len(result1.first().pred.boxes) > 0
        assert len(result2.first().pred.boxes) > 0
