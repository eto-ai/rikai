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
import tempfile

import numpy as np
import pytest
import torch
import torchvision
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import BinaryType, StructField, StructType
from rikai.spark.sql.codegen.fs import FileSystemModel


@pytest.fixture(scope="module")
def yaml_spec():

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Prepare model
        resnet = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            progress=False,
        )
        model_uri = tmp_path / "resnet.pth"
        torch.save(resnet, model_uri)

        spec_yaml = """
version: 1.0
name: resnet
model:
    uri: {}
    flavor: pytorch
schema: struct<boxes:array<array<float>>, scores:array<float>, labels:array<int>>
transforms:
    pre: demoproject.yolo.transform
    post: demoproject.yolo.postprocess
        """.format(
            model_uri
        )  # noqa: E501

        spec_file = tmp_path / "spec.yaml"
        with spec_file.open("w") as fobj:
            fobj.write(spec_yaml)
        yield spec_file


def test_model_codegen_registered(spark: SparkSession):
    spark.sql(
        """CREATE MODEL foo OPTIONS (foo="str",bar=True,max_score=1.23)
         USING 'test://model/a/b/c'"""
    ).count()


def test_yaml_spec(spark: SparkSession, yaml_spec):

    fs_model = FileSystemModel(yaml_spec)
    fs_model.codegen(spark)
    spark.sql("SHOW FUNCTIONS '*resnet*'").show()

    df = spark.createDataFrame(
        [
            Row(
                data=bytearray(
                    np.empty((128, 128, 3), dtype=np.uint8).tobytes()
                )
            )
        ],
        schema=StructType([StructField("data", BinaryType())]),
    )
    df.createOrReplaceTempView("df")
    spark.sql("SELECT resnet(data) as predictions FROM df").show()


def test_yaml_model(spark: SparkSession, yaml_spec):

    spark.sql("CREATE MODEL resnet_m USING 'file://{}'".format(yaml_spec))

    df = spark.createDataFrame(
        [
            Row(
                data=bytearray(
                    np.empty((128, 128, 3), dtype=np.uint8).tobytes()
                )
            )
        ],
        schema=StructType([StructField("data", BinaryType())]),
    )
    df.createOrReplaceTempView("df")

    spark.sql(
        "SELECT ML_PREDICT(resnet_m, data) as predictions FROM df"
    ).show()
