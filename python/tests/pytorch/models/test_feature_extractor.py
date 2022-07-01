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
from typing import Callable

import pytest
import torch
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType, StructField, StructType

from rikai.pytorch.models.feature_extractor import FeatureExtractor
from rikai.types import Image


import torchvision  # noqa

if torchvision.__version__ < "0.12.0":
    pytest.skip("torchvision >= 0.12.0 is required", allow_module_level=True)
import torchvision.transforms as T  # noqa
from torchvision.models import convnext_base, efficientnet_b0, resnet50  # noqa


@pytest.mark.parametrize(
    "model, model_type, dimension",
    [
        (resnet50, "resnet", 2048),
        (efficientnet_b0, "efficientnet_b0", 1280),
        (convnext_base, "convnext", 1024),
    ],
)
def test_torch_classification(
    model: Callable,
    model_type: str,
    dimension: int,
    asset_path: Path,
    spark: SparkSession,
    tmp_path: Path,
):
    m = model(pretrained=True)

    extractor = FeatureExtractor(m, "avgpool")
    uri = str(tmp_path / "extractor.pth")
    torch.save(extractor, uri)

    spark.sql(
        f"""
        CREATE OR REPLACE MODEL test_model
        FLAVOR pytorch
        MODEL_TYPE feature_extractor
        OPTIONS (model_type = '{model_type}')
        USING '{uri}'
        """
    )
    df = spark.sql(
        f"""SELECT ML_PREDICT(
                test_model, image('{asset_path / 'cat.jpg'}')
            ) as embedding"""
    )

    assert df.schema == StructType(
        [StructField("embedding", ArrayType(FloatType()))]
    )
    assert len(df.first().embedding) == dimension


def test_compile_to_torchscript(asset_path: Path, tmp_path: Path):
    resnet = resnet50(pretrained=True)
    extractor = torch.jit.script(FeatureExtractor(resnet, "avgpool"))

    with (tmp_path / "model.pth").open("wb") as fobj:
        torch.jit.save(extractor, fobj)

    loaded = torch.jit.load(str(tmp_path / "model.pth"))

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img = Image(asset_path / "cat.jpg")
    out = loaded(transform(img.to_pil()).unsqueeze(0))
    assert out.shape == (1, 2048, 1, 1)
