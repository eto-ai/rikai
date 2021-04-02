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

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Union

import pandas as pd
import torch
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DataType
from torch.utils.data import DataLoader

from rikai.io import open_uri
from rikai.torch.pandas import PandasDataset


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

    options = {} if spec.options is None else spec.options
    use_gpu = options.get("device", "cpu") == "gpu"
    num_workers = int(options.get("num_workers", 4))
    batch_size = int(options.get("batch_size", 4))

    def torch_inference_udf(
        iter: Iterator[pd.DataFrame],
    ) -> Iterator[pd.DataFrame]:

        with open_uri(spec.uri) as fobj:
            model = torch.load(fobj)
        device = torch.device("cuda" if use_gpu else "cpu")

        model.to(device)
        model.eval()

        with torch.no_grad():
            for series in iter:
                dataset = PandasDataset(series, transform=spec.pre_processing)
                results = []
                for batch in DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                ):
                    predictions = model(batch)
                    if spec.post_processing:
                        predictions = spec.post_processing(predictions)
                    results.extend(predictions)
                yield pd.DataFrame(results)

    return pandas_udf(torch_inference_udf, returnType=spec.schema)
