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
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BinaryType
import numpy as np

from rikai.spark.sql.codegen.base import ModelSpec
from rikai.tf.transforms import convert_tensor
from rikai.torch.pandas import PandasDataset
from rikai.types import Image

import tensorflow as tf

DEFAULT_BATCH_SIZE = 4


__all__ = ["generate_udf", "load_model_from_uri"]


def infer_output_signature(df: pd.DataFrame):
    if df.empty:
        raise ValueError("DataFrame is empty")

    first_row = df.loc[0]
    output_signature = []
    for col in df.columns:
        value = first_row[col]
        if isinstance(value, Image):
            print("Image shape: ", value.to_numpy().shape)
            image_arr = value.to_numpy()
            if len(image_arr.shape) == 2:
                output_signature.append(
                    tf.TensorSpec(shape=(None, None), dtype=tf.uint8, name=col)
                )
            elif len(image_arr.shape) == 3:
                output_signature.append(
                    tf.TensorSpec(
                        shape=(
                            None,
                            None,
                            image_arr.shape[2],
                        ),
                        dtype=tf.uint8,
                        name=col,
                    )
                )
            else:
                raise ValueError(
                    f"Image should only have 2d or 3d shape, got {image_arr.shape}"
                )
        elif isinstance(value, np.ndarray):
            output_signature.append(tf.TensorSpec.from_tensor(value))
        else:
            output_signature.append(tf.TensorSpec.from_tensor(value))
    return tuple(output_signature)


def only_value(data):
    print(data)
    return list(data.values())


def generate_udf(spec: ModelSpec):
    """Construct a UDF to run Tensorflow (Karas) Model"""

    batch_size = int(spec.options.get("batch_size", DEFAULT_BATCH_SIZE))

    def tf_inference_udf(
        iter: Iterator[pd.DataFrame],
    ) -> Iterator[pd.Series]:
        import tensorflow as tf

        model = spec.load_model()
        signature = None
        for df in iter:
            ds = PandasDataset(df)

            if signature is None:
                signature = infer_output_signature(df)

            print("SIGANTURES: ", signature)
            data = tf.data.Dataset.from_generator(
                ds, output_signature=signature
            )

            data = data.batch(1)
            if spec.pre_processing:
                data = data.map(spec.pre_processing)

            print(list(data.take(1)))
            for elem in data:
                raw_predictions = model(elem)
                print(raw_predictions)
            # model.signatures["serving_default"](data)
            return [1] * len(df)

    return pandas_udf(tf_inference_udf, returnType=BinaryType())


def load_model_from_uri(uri: str):
    from tensorflow import keras

    return keras.models.load_model(uri)
