#  Copyright 2020 Rikai Authors
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

# Standard Library
import os
from pathlib import Path

# Third Party
import numpy as np
from pyspark.sql import Row, SparkSession

# Rikai
from rikai.numpy import wrap
from rikai.torch import DataLoader
from rikai.types import Box2d, Image


def test_load_dataset(spark: SparkSession, tmp_path: Path):
    dataset_dir = tmp_path / "features"
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir(parents=True)

    expected = []
    data = []
    for i in range(1000):
        image_data = np.random.randint(0, 128, size=(128, 128), dtype=np.uint8)
        image_uri = asset_dir / f"{i}.png"

        array = wrap(np.random.random_sample((3, 4)))
        data.append(
            {
                "id": i,
                "array": array,
                "image": Image.from_array(image_data, image_uri),
            }
        )
        expected.append({"id": i, "array": array, "image": image_data})
    df = spark.createDataFrame(data)

    df.write.mode("overwrite").format("rikai").save(str(dataset_dir))

    loader = DataLoader(dataset_dir, batch_size=8)
    actual = []
    for examples in loader:
        # print(examples)
        assert len(examples) == 8
        actual.extend(examples)

    actual = sorted(actual, key=lambda x: x["id"])
    assert len(actual) == 1000
    for expect, act in zip(expected, actual):
        assert np.array_equal(expect["array"], act["array"])
        assert np.array_equal(expect["image"], act["image"])


def test_coco_dataset(
    spark: SparkSession,
    tmp_path: Path,
):
    dataset_dir = tmp_path / "features"
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir(parents=True)
    data = []
    for i in range(10):
        image_data = np.random.randint(0, 128, size=(128, 128), dtype=np.uint8)
        image_uri = asset_dir / f"{i}.png"

        data.append(
            Row(
                image_id=i,
                split="train",
                image=Image.from_array(image_data, image_uri),
                annotations=[
                    Row(
                        category_id=123,
                        category_text="car",
                        bbox=Box2d(1, 2, 3, 4),
                    ),
                    Row(
                        category_id=234,
                        category_text="dog",
                        bbox=Box2d(1, 2, 3, 4),
                    ),
                ],
            )
        )

    spark.createDataFrame(data).write.mode("overwrite").format("rikai").save(
        str(dataset_dir)
    )

    loader = DataLoader(dataset_dir, batch_size=1)
    example = next(iter(loader))
    assert isinstance(example, list)
    assert 1 == len(example)
    assert 2 == len(example[0]["annotations"])
    assert np.array_equal(
        np.array([1, 2, 3, 4]), example[0]["annotations"][0]["bbox"]
    ), f"Actual annotations: {example[0]['annotations'][0]['bbox']}"
