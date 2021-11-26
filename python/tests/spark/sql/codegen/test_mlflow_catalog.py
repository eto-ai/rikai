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

import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression


def test_list_models(spark_with_mlflow, mlflow_client):
    print('test_list_models')
    spark = spark_with_mlflow
    x = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(x, np.array([1, 2])) + 3
    model = LinearRegression()
    with mlflow.start_run():
        model.fit(x, y)

    tracking_uri = mlflow.get_tracking_uri()
    print("Tracking URI", tracking_uri)
    spark.sql("SHOW MODELS").show()
