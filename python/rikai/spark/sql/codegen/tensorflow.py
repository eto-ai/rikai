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

from rikai.spark.sql.codegen.base import ModelSpec

DEFAULT_BATCH_SIZE = 4


def generate_udf(spec: ModelSpec):
    """Construct a UDF to run Tensorflow (Karas) Model"""

    batch_size = int(spec.options.get("batch_size", DEFAULT_BATCH_SIZE))

    def tf_inference_udf(
        iter: Iterator[pd.DataFrame],
    ) -> Iterator[pd.Series]:
        import tensorflow as tf

        model = spec.load_model()

        for df in iter:
            print(df)
            data = tf.data.Dataset.from_tensor_slices(df)
            data = data.batch(batch_size).map(spec.pre_processing)

            print(data)
            raw_predictions = model(data)
            return [1] * len(df)

    return pandas_udf(tf_inference_udf, returnType=BinaryType())


def load_model_from_uri(uri: str):
    from tensorflow import keras

    return keras.models.load_model(uri)
