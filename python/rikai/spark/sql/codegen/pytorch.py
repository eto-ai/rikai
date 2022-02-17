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
from rikai.pytorch.pandas import PandasDataset
from rikai.spark.sql.model import AnonymousModelType, ModelSpec

DEFAULT_NUM_WORKERS = 8
DEFAULT_BATCH_SIZE = 4

_pickler = CloudPickleSerializer()


def move_tensor_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_tensor_to_device(elem, device) for elem in data]
    # Do nothing
    return data


def _generate(payload: ModelSpec, is_udf: bool = True):
    """Construct a UDF to run pytorch model.

    Parameters
    ----------
    payload : ModelSpec
        the model specifications object

    Returns
    -------
    A Spark Pandas UDF.
    """
    model = payload.model_type
    default_device = "gpu" if torch.cuda.is_available() else "cpu"
    options = payload.options
    use_gpu = options.get("device", default_device) == "gpu"
    num_workers = int(
        options.get("num_workers", min(os.cpu_count(), DEFAULT_NUM_WORKERS))
    )
    batch_size = int(options.get("batch_size", DEFAULT_BATCH_SIZE))

    return_type = Iterator[pd.Series]

    def torch_inference_udf(
        iter: Iterator[pd.DataFrame],
    ) -> return_type:
        device = torch.device("cuda" if use_gpu else "cpu")
        model.load_model(payload, device=device)
        if isinstance(model, AnonymousModelType):
            # We will remove them after AnonymousModelType deprecation
            model.model.eval()
            model.model.to(device)

        try:
            with torch.no_grad():
                for series in iter:
                    dataset = PandasDataset(
                        series,
                        transform=model.transform(),
                        unpickle=is_udf,
                        use_pil=True,
                    )
                    results = []
                    for batch in DataLoader(
                        dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                    ):
                        batch = move_tensor_to_device(batch, device)
                        predictions = model(batch)
                        bin_predictions = [
                            _pickler.dumps(p) if is_udf else p
                            for p in predictions
                        ]
                        results.extend(bin_predictions)
                    yield pd.Series(results)
        finally:
            if use_gpu:
                model.release()

    if is_udf:
        return pandas_udf(torch_inference_udf, returnType=BinaryType())
    else:
        return torch_inference_udf


def generate_inference_func(payload: ModelSpec):
    return _generate(payload, False)


def generate_udf(payload: ModelSpec):
    return _generate(payload, True)


def load_model_from_uri(uri: str):
    with open_uri(uri) as fobj:
        return torch.load(fobj)
