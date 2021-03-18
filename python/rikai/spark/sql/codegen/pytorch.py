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
from torch.utils.data import DataLoader
from pyspark.sql.types import DataType
from pyspark.sql.functions import pandas_udf

from rikai.io import open_uri
from rikai.torch.pandas import PandasDataset


__all__ = ["runner"]


def runner(
    model_uri: Union[str, Path],
    schema: DataType,
    pre_processing: Optional[Callable] = None,
    post_processing: Optional[Callable] = None,
    options: Optional[Dict[str, str]] = None,
):
    options = {} if options is None else options
    use_gpu = options.get("device", "cpu") == "gpu"
    num_workers = int(options.get("num_workers", 4))

    def torch_inference(
        iter: Iterator[pd.DataFrame],
    ) -> Iterator[pd.DataFrame]:
        with open_uri(model_uri) as model_obj:
            model = torch.load(model_obj)
        device = torch.device("cuda" if use_gpu else "cpu")

        model.to(device)
        model.eval()

        with torch.no_grad():
            for series in iter:
                dataset = PandasDataset(series, transform=pre_processing)
                batch_result = {"boxes": [], "scores": [], "labels": []}
                for batch in DataLoader(dataset, num_workers=num_workers):
                    predictions = model(batch)
                    print(predictions)
                    if post_processing:
                        predictions = post_processing(predictions)
                    for p in predictions:
                        batch_result["boxes"].append(p["boxes"].tolist())
                        batch_result["scores"].append(p["scores"].tolist())
                        batch_result["labels"].append(p["labels"].tolist())
                yield pd.DataFrame(batch_result)

    return pandas_udf(torch_inference, schema)
