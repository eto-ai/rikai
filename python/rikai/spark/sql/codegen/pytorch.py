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
from pyspark.sql.functions import pandas_udf
from torch.utils.data import DataLoader

from rikai.io import open_uri
from rikai.torch.pandas import PandasDataset

DEFAULT_NUM_WORKERS = 8
DEFAULT_BATCH_SIZE = 4


def collate_fn(items):
    """Collate fn that split data and errors."""
    data = []
    errors = []
    for item in items:
        if "data" in item:
            data.append(item["data"])
            errors.append(None)
        else:
            errors.append(item)
    return torch.stack(data), errors


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
    use_gpu = spec.options.get("device", "cpu") == "gpu"
    num_workers = int(
        spec.options.get(
            "num_workers", min(os.cpu_count(), DEFAULT_NUM_WORKERS)
        )
    )
    batch_size = int(spec.options.get("batch_size", DEFAULT_BATCH_SIZE))
    with_error = bool(spec.options.get("with_error", False))

    def torch_inference_udf(
        df_iter: Iterator[pd.DataFrame],
    ) -> Iterator[pd.DataFrame]:
        device = torch.device("cuda" if use_gpu else "cpu")
        model = spec.load_model()
        model.to(device)
        model.eval()

        with torch.no_grad():
            for series in df_iter:
                dataset = PandasDataset(series, transform=spec.pre_processing)
                results = []
                for batch, errors in DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                ):
                    batch = batch.to(device)
                    predictions = model(batch)
                    if spec.post_processing:
                        predictions = spec.post_processing(predictions)
                    ret = errors

                    pred_iter = iter(predictions)
                    result = []
                    for idx, val in enumerate(ret):
                        if val is None:
                            result.append(next(pred_iter))
                        else:
                            if with_error:
                                result.append(
                                    {"_error": val["exception"].message}
                                )
                            else:
                                result.append(None)
                    results.extend(result)
                yield pd.DataFrame(results)

    schema = spec.schema
    if with_error:
        print(schema)
    return pandas_udf(torch_inference_udf, returnType=spec.schema)


def load_model_from_uri(uri: str):
    with open_uri(uri) as fobj:
        return torch.load(fobj)
