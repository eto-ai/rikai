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

import os
from typing import Iterator

import pandas as pd
import torch
from pyspark.serializers import CloudPickleSerializer
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BinaryType
from torch.utils.data import DataLoader

from rikai.io import open_uri
from rikai.torch.pandas import PandasDataset

DEFAULT_NUM_WORKERS = 8
DEFAULT_BATCH_SIZE = 4

_pickler = CloudPickleSerializer()


def generate_udf(spec: "rikai.spark.sql.codegen.base.ModelSpec"):
    """Construct a UDF to run pytorch model.

    Parameters
    ----------
    spec : ModelSpec
        the model specifications object

    Returns
    -------
    A Spark Pandas UDF.
    """
    default_device = "gpu" if torch.cuda.is_available() else "cpu"
    use_gpu = spec.options.get("device", default_device) == "gpu"
    num_workers = int(
        spec.options.get(
            "num_workers", min(os.cpu_count(), DEFAULT_NUM_WORKERS)
        )
    )
    batch_size = int(spec.options.get("batch_size", DEFAULT_BATCH_SIZE))

    return_type = Iterator[pd.Series]

    def torch_inference_udf(
        iter: Iterator[pd.DataFrame],
    ) -> return_type:
        device = torch.device("cuda" if use_gpu else "cpu")
        model = spec.load_model()
        model.to(device)
        model.eval()

        with torch.no_grad():
            for series in iter:
                dataset = PandasDataset(
                    series,
                    transform=spec.pre_processing,
                    unpickle=True,
                    use_pil=True,
                )
                results = []
                for batch in DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                ):
                    if isinstance(batch, torch.Tensor):
                        batch = batch.to(device)
                    predictions = model(batch)
                    if spec.post_processing:
                        predictions = spec.post_processing(predictions)
                    bin_predictions = [_pickler.dumps(p) for p in predictions]
                    results.extend(bin_predictions)
                yield pd.Series(results)

        # Release GPU memory
        # https://blog.paperspace.com/pytorch-memory-multi-gpu-debugging/?ref=tfrecipes
        if use_gpu:
            model = model.cpu()
            del model
            torch.cuda.empty_cache()

    return pandas_udf(torch_inference_udf, returnType=BinaryType())


def load_model_from_uri(uri: str):
    with open_uri(uri) as fobj:
        return torch.load(fobj)
