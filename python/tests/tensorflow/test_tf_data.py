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

import numpy as np
import tensorflow as tf
from pyspark.sql import Row

from rikai.tensorflow.data import from_rikai
from rikai.types import Image


def test_tf_dataset(spark, tmp_path):
    dataset_dir = tmp_path / "data"
    data = []
    for i in range(100):
        image_data = np.random.randint(
            0, 128, size=(128, 128, 3), dtype=np.uint8
        )
        image = Image.from_array(image_data)

        data.append(Row(id=i, image=image))
    df = spark.createDataFrame(data)
    df.write.format("rikai").save(str(dataset_dir))

    dataset = from_rikai(
        dataset_dir,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.uint8),
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        ),
    )

    assert len(list(dataset)) == 100
    for row, (id, img) in zip(data, dataset):
        assert id.get_shape() == ()
        assert tf.constant(row.id, dtype=tf.uint8) == id
        assert img.get_shape() == (128, 128, 3)
        assert np.array_equal(row.image.to_numpy(), img.numpy())
