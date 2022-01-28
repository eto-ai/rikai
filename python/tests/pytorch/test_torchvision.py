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
#  limitations under the License.

"""Test torchvision compabilities"""
import random
from pathlib import Path
from typing import Union

import numpy as np
import torch
from pyspark.sql import DataFrame, Row, SparkSession
from torchvision import transforms

from rikai.pytorch.vision import Dataset
from rikai.types import Image


def test_vision_dataset(spark: SparkSession, tmp_path: Path):
    df = _create_dataframe(tmp_path, spark)
    df.show()
    dataset_dir = str(tmp_path / "data")
    df.write.mode("overwrite").format("rikai").save(dataset_dir)

    dataset = _create_dataset(dataset_dir)
    dataset_from_df = _create_dataset(df)
    img, target = next(iter(dataset))

    assert df.first()["label"] == target
    assert torch.equal(dataset.transform(df.first()["image"].to_pil()), img)
    assert isinstance(img, torch.Tensor)
    assert len(list(dataset)) == 100

    for (img1, target1), (img2, target2) in zip(
        iter(dataset), iter(dataset_from_df)
    ):
        assert torch.equal(img1, img2)
        assert target1 == target2


def _create_dataframe(df_path: Path, spark: SparkSession):
    asset_dir = df_path / "assets"
    asset_dir.mkdir(parents=True)

    data = []
    for i in range(100):
        image_data = np.random.randint(
            0, 128, size=(64, 64, 3), dtype=np.uint8
        )
        image_uri = asset_dir / f"{i}.png"
        data.append(
            Row(
                id=i,
                image=Image.from_array(image_data, image_uri),
                label=random.choice(["cat", "dog", "duck", "bird"]),
            )
        )
    return spark.createDataFrame(data)


def _create_dataset(data: Union[str, Path, DataFrame]):
    transform = transforms.Compose(
        transforms=[
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return Dataset(data, "image", "label", transform=transform)
