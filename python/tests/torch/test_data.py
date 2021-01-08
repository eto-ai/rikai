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
from pickle import encode_long

# Third Party
from PIL import Image as PILImage
import PIL
import numpy as np
from numpy.lib.arraysetops import isin
from pyspark.sql import Row

# Rikai
from rikai.numpy import wrap
from rikai.vision import Image, BBox
from rikai.testing.spark import SparkTestCase
from rikai.torch import DataLoader


class TorchDataLoaderTest(SparkTestCase):
    def test_load_dataset(self):
        dataset_dir = os.path.join(self.test_dir, "features")
        asset_dir = os.path.join(self.test_dir, "assets")
        os.makedirs(asset_dir)

        expected = []
        data = []
        for i in range(1000):
            image_data = np.random.randint(0, 128, size=(128, 128), dtype=np.uint8)
            image_uri = os.path.join(asset_dir, f"{i}.png")
            PILImage.fromarray(image_data).save(image_uri)

            array = wrap(np.random.random_sample((3, 4)))
            data.append(
                {
                    "id": i,
                    "array": array,
                    "image": Image(image_uri),
                }
            )
            expected.append({"id": i, "array": array, "image": image_data})
        df = self.spark.createDataFrame(data)

        df.write.mode("overwrite").format("rikai").save(dataset_dir)

        loader = DataLoader(dataset_dir, batch_size=8)
        actual = []
        for examples in loader:
            # print(examples)
            self.assertEqual(8, len(examples))
            actual.extend(examples)

        actual = sorted(actual, key=lambda x: x["id"])
        self.assertEqual(1000, len(actual))
        for expect, act in zip(expected, actual):
            self.assertTrue(np.array_equal(expect["array"], act["array"]))
            self.assertTrue(np.array_equal(expect["image"], act["image"]))

    def test_coco_dataset(self):
        dataset_dir = os.path.join(self.test_dir, "features")
        asset_dir = os.path.join(self.test_dir, "assets")
        os.makedirs(asset_dir)
        data = []
        for i in range(10):
            image_data = np.random.randint(0, 128, size=(128, 128), dtype=np.uint8)
            image_uri = os.path.join(asset_dir, f"{i}.png")
            PILImage.fromarray(image_data).save(image_uri)

            data.append(
                Row(
                    image_id=i,
                    split="train",
                    image=Image(image_uri),
                    annotations=[
                        Row(
                            category_id=123, category_text="car", bbox=BBox(1, 2, 3, 4)
                        ),
                        Row(
                            category_id=234, category_text="dog", bbox=BBox(1, 2, 3, 4)
                        ),
                    ],
                )
            )

        self.spark.createDataFrame(data).write.mode("overwrite").format("rikai").save(
            dataset_dir
        )

        loader = DataLoader(dataset_dir, batch_size=1)
        example = next(iter(loader))
        self.assertTrue(isinstance(example, list))
        self.assertEqual(1, len(example))
        self.assertEqual(2, len(example[0]["annotations"]))
        self.assertTrue(
            np.array_equal(np.array([1, 2, 3, 4]), example[0]["annotations"][0]["bbox"])
        )
