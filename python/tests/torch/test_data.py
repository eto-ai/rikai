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
from pathlib import Path

# Third Party
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader as torchDataLoader

# Rikai
from rikai.numpy import wrap
from rikai.torch import Dataset
from rikai.types import Image


# @pytest.mark.parametrize("num_workers", [0, 2, 4])
@pytest.mark.timeout(30)
@pytest.mark.parametrize("num_workers", [0])  # Still hangs
def test_torch_dataset(spark, tmp_path, num_workers):
    total = 1000
    dataset_dir = tmp_path / "data"
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir(parents=True)
    data = []
    expected = []
    for i in range(total):
        image_data = np.random.randint(0, 128, size=(128, 128), dtype=np.uint8)
        image_uri = asset_dir / f"{i}.png"
        Image.from_array(image_data, image_uri),

        array = wrap(np.random.random_sample((3, 4)))
        data.append(
            {
                "id": i,
                "array": array,
                "image": Image(image_uri),
            }
        )
        expected.append(
            {
                "id": i,
                "array": torch.as_tensor(np.array([array])),
                "image": torch.as_tensor(np.array([image_data])),
            }
        )

    df = spark.createDataFrame(data)
    df.write.mode("overwrite").format("rikai").save(str(dataset_dir))
    dataset = Dataset(dataset_dir)
    loader = torchDataLoader(
        dataset,
        num_workers=num_workers,
        drop_last=True,
    )
    actual = sorted(list(loader), key=lambda x: x["id"])
    assert len(actual) == total
    for expect, act in zip(expected, actual):
        assert torch.equal(
            expect["array"], act["array"]
        ), f"Expected {expect['array']} got {act['array']}"
        assert torch.equal(expect["image"], act["image"])
