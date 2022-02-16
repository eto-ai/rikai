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

from typing import Iterator

import pandas as pd
import tensorflow as tf
from pyspark.serializers import CloudPickleSerializer
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BinaryType

from rikai.pytorch.pandas import PandasDataset
from rikai.spark.sql.codegen.base import ModelSpec
from rikai.types import Image

DEFAULT_BATCH_SIZE = 4


__all__ = ["generate_udf", "load_model_from_uri"]

_pickler = CloudPickleSerializer()


def infer_output_signature(blob, is_udf: bool):
    if is_udf:
        row = _pickler.loads(blob)
    else:
        row = blob

    if isinstance(row, Image):
        image_arr = row.to_numpy()
        if len(image_arr.shape) == 2:
            return tf.TensorSpec(shape=(None, None), dtype=tf.uint8)
        elif len(image_arr.shape) == 3:
            return tf.TensorSpec(
                shape=(
                    None,
                    None,
                    image_arr.shape[2],
                ),
                dtype=tf.uint8,
            )
        else:
            raise ValueError(
                f"Image should only have 2d or 3d shape, got {image_arr.shape}"
            )
    else:
        return tf.TensorSpec.from_tensor(row)


def _generate(spec: ModelSpec, is_udf: bool = True):
    """Construct a UDF to run Tensorflow (Karas) Model"""

    batch_size = int(spec.options.get("batch_size", DEFAULT_BATCH_SIZE))

    def tf_inference_udf(
        iter: Iterator[pd.DataFrame],
    ) -> Iterator[pd.Series]:

        model = spec.load_model()

        signature = None
        for df in iter:
            if signature is None:
                signature = infer_output_signature(df.iloc[0], is_udf)

            ds = PandasDataset(df, unpickle=is_udf, use_pil=True)
            data = tf.data.Dataset.from_generator(
                ds,
                output_signature=tf.TensorSpec(
                    shape=(None, None, 3), dtype=tf.uint8, name="input_tensor"
                ),
            )

            if spec.pre_processing:
                data = data.map(spec.pre_processing)
            data = data.batch(batch_size)

            predictions = []
            for batch in data:
                raw_predictions = model(batch)
                predictions.extend(spec.post_processing(raw_predictions))
            results = [_pickler.dumps(p) if is_udf else p for p in predictions]
            yield pd.Series(results)

    if is_udf:
        return pandas_udf(tf_inference_udf, returnType=BinaryType())
    else:
        return tf_inference_udf


def generate_inference_func(payload: ModelSpec):
    return _generate(payload, False)


def generate_udf(payload: ModelSpec):
    return _generate(payload, True)


def load_model_from_uri(uri: str):
    from tensorflow import keras

    return keras.models.load_model(uri)
