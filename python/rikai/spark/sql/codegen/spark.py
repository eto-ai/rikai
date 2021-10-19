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
from typing import Iterator

import pandas as pd
from pyspark.sql.functions import pandas_udf

def generate_udf(spec: "rikai.spark.sql.codegen.base.ModelSpec"):
    """Construct a UDF to run sparkml model.

    Parameters
    ----------
    spec : ModelSpec
        the model specifications object

    Returns
    -------
    A Spark ML UDF.
    """

    def spark_ml_udf(
            iter: Iterator[pd.DataFrame],
    ) -> Iterator[pd.DataFrame]:
        model = spec.load_model()
        for old_frame in list(iter):
            new_frame = model.transform(old_frame)
            yield new_frame

    return pandas_udf(spark_ml_udf, returnType=spec.schema)
