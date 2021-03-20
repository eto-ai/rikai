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
from typing import Callable, Dict, Iterator, Optional, Union

import pandas as pd
import torch
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DataType
from torch.utils.data import DataLoader

from rikai.io import open_uri
from rikai.torch.pandas import PandasDataset


def generate_udf(
    model_uri: Union[str, Path],
    schema: DataType,
    options: Optional[Dict[str, str]] = None,
    pre_processing: Optional[Callable] = None,
    post_processing: Optional[Callable] = None,
):
    """Construct a UDF to run pytorch model.

    Parameters
    ----------
    model_uri : str
        The URI pointed to a model.
    schema : DataType
        Return type of the model inference function.
    options : Dict[str, str], optional
        Runtime options

    Returns
    -------
    A Spark Pandas UDF.
    """
    options = {} if options is None else options
    use_gpu = options.get("device", "cpu") == "gpu"
    num_workers = int(options.get("num_workers", 4))
    batch_size = int(options.get("batch_size", 4))

    def torch_inference_udf(
        iter: Iterator[pd.DataFrame],
    ) -> Iterator[pd.DataFrame]:

        with open_uri(model_uri) as fobj:
            model = torch.load(fobj)
        device = torch.device("cuda" if use_gpu else "cpu")

        model.to(device)
        model.eval()

        with torch.no_grad():
            for series in iter:
                dataset = PandasDataset(series, transform=pre_processing)
                results = []
                for batch in DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                ):
                    predictions = model(batch)
                    if post_processing:
                        predictions = post_processing(predictions)
                    results.extend(predictions)
                yield pd.DataFrame(results)

    return pandas_udf(torch_inference_udf, returnType=schema)
