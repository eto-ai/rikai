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

import numpy as np
from pyspark.sql import SparkSession, Row
from torchvision import transforms

from rikai.torch.vision import Dataset
from rikai.types import Box2d, Image


def test_vision_dataset(spark: SparkSession, tmp_path: Path):
    dataset_dir = tmp_path / "data"
    asset_dir = tmp_path / "assets"
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
    df = spark.createDataFrame(data)
    df.show()

    df.write.mode("overwrite").format("rikai").save(str(dataset_dir))

    transform = transforms.Compose(
        transforms=[
            transforms.CenterCrop(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    dataset = Dataset(dataset_dir, "image", "label", transform=transform)
    first = next(iter(dataset))
    print(first)
