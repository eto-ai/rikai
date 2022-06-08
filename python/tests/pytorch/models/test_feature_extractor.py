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

import torch
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType, StructField, StructType
from torchvision.models import resnet50, efficientnet_b0

from rikai.pytorch.models.feature_extractor import FeatureExtractor


def test_resnet_features(
    asset_path: Path, spark: SparkSession, tmp_path: Path
):
    resnet = resnet50(pretrained=True)

    extractor = FeatureExtractor(resnet, "avgpool")
    uri = str(tmp_path / "extractor.pth")
    torch.save(extractor, uri)

    spark.sql(
        f"""
    CREATE OR REPLACE MODEL resnet_features
    FLAVOR pytorch
    MODEL_TYPE feature_extractor
    OPTIONS (model_type = 'resnet')
    USING '{uri}'
    """
    )
    df = spark.sql(
        f"""SELECT ML_PREDICT(
            resnet_features, image('{asset_path / 'cat.jpg'}')
        ) as embedding"""
    )

    assert df.schema == StructType(
        [StructField("embedding", ArrayType(FloatType()))]
    )
    assert len(df.first().embedding) == 2048


def test_efficientnet_features(
    tmp_path: Path, spark: SparkSession, asset_path: Path
):
    effnet = efficientnet_b0(pretrained=True)

    extractor = FeatureExtractor(effnet, "avgpool")
    uri = str(tmp_path / "extractor.pth")
    torch.save(extractor, uri)

    spark.sql(
        f"""
        CREATE OR REPLACE MODEL effnet_features
        FLAVOR pytorch
        MODEL_TYPE feature_extractor
        OPTIONS (model_type = 'efficientnet_b0')
        USING '{uri}'
        """
    )
    df = spark.sql(
        f"""SELECT ML_PREDICT(
                effnet_features, image('{asset_path / 'cat.jpg'}')
            ) as embedding"""
    )

    assert df.schema == StructType(
        [StructField("embedding", ArrayType(FloatType()))]
    )
    assert len(df.first().embedding) == 1280
