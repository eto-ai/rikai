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


from typing import Callable, Dict, Iterator, Optional

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DataType

from rikai.io import open_uri


def pytorch_runner(
    model_uri: str,
    schema: DataType,
    pre_processing: Optional[Callable] = None,
    post_processing: Optional[Callable] = None,
    options: Optional[Dict[str, str]] = None,
):
    """Construct a UDF to run pytorch model."""
    if pre_processing is None:
        pre_processing = lambda x: x
    if post_processing is None:
        post_processing = lambda x: x

    def torch_inference_udf(
        iter: Iterator[pd.DataFrame],
    ) -> Iterator[pd.DataFrame]:
        import torch

        with open_uri(model_uri) as fobj:
            model = torch.load(fobj)
        model.eval()

        for series in iter:
            yield series

    return pandas_udf(torch_inference_udf, returnType=schema)
