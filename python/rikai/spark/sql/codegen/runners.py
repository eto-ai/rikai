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
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, Optional
import json

import numpy as np
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import (
    ArrayType,
    DataType,
    IntegerType,
    StructField,
    StructType,
    StringType,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from rikai.io import open_uri


@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


class _Dataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform: Callable):
        self.data: pd.DataFrame = data
        self.transform = transform

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index):
        row = self.data.iloc[index]
        if self.transform:
            row = self.transform(row)
        return row


def to_numpy(b):
    return np.frombuffer(b, dtype=np.uint8).reshape(128, 128, 3)


def pytorch_runner(
    yaml_uri: str,
    model_uri: str,
    schema: DataType,
    pre_processing: Optional[Callable] = None,
    post_processing: Optional[Callable] = None,
    options: Optional[Dict[str, str]] = None,
):
    """Construct a UDF to run pytor ch model."""
    options = {} if options is None else options
    use_gpu = options.get("device", "cpu") == "gpu"

    def torch_inference_udf(
        iter: Iterator[pd.DataFrame],
    ) -> Iterator[pd.DataFrame]:
        import torch

        with open_uri(
            os.path.join(os.path.dirname(yaml_uri), model_uri)
        ) as fobj:
            model = torch.load(fobj)
        device = torch.device("cuda" if use_gpu else "cpu")

        model.to(device)
        model.eval()

        transform = T.Compose(
            [
                to_numpy,
                T.functional.to_pil_image,
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
            ]
        )

        with torch.no_grad():
            for series in iter:
                print("Series[0] = ", series.iloc[0])
                dataset = _Dataset(series, transform=transform)
                batch_result = {"boxes": []}
                for batch in DataLoader(dataset, num_workers=4):
                    # print(batch)
                    print(model(batch))
                    predictions = model(batch)
                    print("GOT PREDICTIONS ", predictions)
                    if post_processing:
                        predictions = post_processing(predictions)
                    boxes = []
                    for p in predictions:
                        boxes.append(p["boxes"].tolist())
                    batch_result["boxes"].append(boxes)
                yield pd.DataFrame(batch_result)

    return pandas_udf(
        torch_inference_udf,
        returnType=StructType(
            [StructField("boxes", ArrayType(ArrayType(IntegerType())))]
        ),
    )
