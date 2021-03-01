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

from pyspark.sql.session import SparkSession

from rikai.spark.sql.codegen.fs import FileSystemModel


def test_model_codegen_registered(spark: SparkSession):
    spark.sql(
        """CREATE MODEL foo OPTIONS (foo="str",bar=True,max_score=1.23)
         USING 'test://model/a/b/c'"""
    ).count()


def test_yaml_spec(spark: SparkSession, tmp_path: Path):
    spec_yaml = """
version: 1.0
name: yolo_test
model:
  uri: yolo.pt
  flavor: pytorch
schema: struct<box:box2d, score:float, class:int>
transforms:
  pre: demoproject.yolo.transform
  post: demoproject.yolo.postprocess

    """
    spec_file = tmp_path / "spec.yaml"
    with spec_file.open("w") as fobj:
        fobj.write(spec_yaml)

    fs_model = FileSystemModel(spec_file)
    fs_model.codegen(spark)
