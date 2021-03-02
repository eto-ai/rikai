#  Copyright 2021 Rikai Authors
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

from pathlib import Path

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import BinaryType, StructField, StructType
import torchvision
import torch
import numpy as np

from rikai.spark.sql.codegen.fs import FileSystemModel
from rikai.numpy import wrap


def test_model_codegen_registered(spark: SparkSession):
    spark.sql(
        """CREATE MODEL foo OPTIONS (foo="str",bar=True,max_score=1.23)
         USING 'test://model/a/b/c'"""
    ).count()


def test_yaml_spec(spark: SparkSession, tmp_path: Path):
    spec_yaml = """
version: 1.0
name: resnet
model:
  uri: resnet.pth
  flavor: pytorch
schema: struct<box:box2d, score:float, class:int>
transforms:
  pre: demoproject.yolo.transform
  post: demoproject.yolo.postprocess

    """
    spec_file = tmp_path / "spec.yaml"
    with spec_file.open("w") as fobj:
        fobj.write(spec_yaml)
    # Prepare model
    resnet = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    torch.save(resnet, (tmp_path / "resnet.pth"))

    fs_model = FileSystemModel(spec_file)
    fs_model.codegen(spark)
    spark.sql("SHOW FUNCTIONS '*resnet*'").show()

    df = spark.createDataFrame(
        [
            Row(
                data=bytearray(
                    np.empty((3, 128, 128), dtype=np.uint8).tobytes()
                )
            )
        ],
        schema=StructType([StructField("data", BinaryType())]),
    )
    df.createOrReplaceTempView("df")
    spark.sql("SELECT resnet(data) FROM df").show()
