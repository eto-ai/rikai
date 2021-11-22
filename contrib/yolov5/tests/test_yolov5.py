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
import pathlib
import urllib

import mlflow
import torch
import yolov5
from pyspark.sql.session import SparkSession

import rikai
from rikai.contrib.yolov5.model import RikaiYolov5Model
from rikai.contrib.yolov5.transforms import OUTPUT_SCHEMA


def test_yolov5(spark: SparkSession):
    mlflow_tracking_uri = "sqlite:///mlruns.db"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    registered_model_name = f"yolov5s-model"
    mlflow.set_experiment(registered_model_name)

    with mlflow.start_run():
        ####
        # Part 1: Train the model and register it on MLflow
        ####
        pretrained = "yolov5s.pt"
        url = "https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt"
        if os.path.exists(pretrained):
            urllib.request.urlretrieve(url, pretrained)
        model = RikaiYolov5Model(yolov5.load(pretrained))
        pre = "rikai.contrib.yolov5.transforms.pre_processing"
        post = "rikai.contrib.yolov5.transforms.post_processing"

        # Rikai's logger adds output_schema, pre_pocessing, and post_processing as additional
        # arguments and automatically adds the flavor / rikai model spec version
        rikai.mlflow.pytorch.log_model(
            model,
            "model",
            OUTPUT_SCHEMA,
            pre_processing=pre,
            post_processing=post,
            registered_model_name=registered_model_name,
        )

        ####
        # Part 2: create the model using the registered MLflow uri
        ####
        if torch.cuda.is_available():
            print("Using GPU\n")
            device = "gpu"
        else:
            print("Using CPU\n")
            device = "cpu"

        spark.conf.set(
            "rikai.sql.ml.registry.mlflow.tracking_uri", mlflow_tracking_uri
        )
        spark.sql(
            f"""
        CREATE MODEL mlflow_yolov5_m OPTIONS (device='{device}') USING 'mlflow:///{registered_model_name}';
        """
        )

        ####
        # Part 3: predict using the registered Rikai model
        ####
        spark.sql("show models").show(1, vertical=False, truncate=False)

        work_dir = pathlib.Path().absolute().parent.parent
        result = spark.sql(
            f"""
        select ML_PREDICT(mlflow_yolov5_m, '{work_dir}/python/tests/assets/test_image.jpg')
        """
        )

        result.printSchema()
        result.show(1, vertical=False, truncate=False)
