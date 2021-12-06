#  Copyright (c) 2021 Rikai Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import getpass

import mlflow
import numpy as np
from pyspark.sql import SparkSession
from sklearn.linear_model import LinearRegression

import rikai


def test_linear_regression(spark: SparkSession):
    mlflow_tracking_uri = "sqlite:///mlruns.db"
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # enable autologging
    mlflow.sklearn.autolog()

    # prepare training data
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3

    # prepare evaluation data
    X_eval = np.array([[3, 3], [3, 4]])
    y_eval = np.dot(X_eval, np.array([1, 2])) + 3

    # train a model
    model = LinearRegression()
    with mlflow.start_run() as run:
        ####
        # Part 1: Train the model and register it on MLflow
        ####
        model.fit(X, y)
        metrics = mlflow.sklearn.eval_and_log_metrics(
            model, X_eval, y_eval, prefix="val_"
        )

        schema = "float"
        registered_model_name = f"{getpass.getuser()}_sklearn_lr"
        rikai.mlflow.sklearn.log_model(
            model, "model", schema, registered_model_name=registered_model_name
        )

        ####
        # Part 2: create the model using the registered MLflow uri
        ####
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
        spark.conf.set(
            "rikai.sql.ml.registry.mlflow.tracking_uri", mlflow_tracking_uri
        )
        spark.sql(
            f"""
        CREATE MODEL mlflow_sklearn_m USING 'mlflow:///{registered_model_name}';
        """
        )

        ####
        # Part 3: predict using the registered Rikai model
        ####
        spark.sql("show models").show(1, vertical=False, truncate=False)

        df = spark.range(100).selectExpr("id as x0", "id+1 as x1")
        df.createOrReplaceTempView("tbl_X")

        result = spark.sql(
            f"""
        select ML_PREDICT(mlflow_sklearn_m, array(x0, x1)) from tbl_X
        """
        )

        result.printSchema()
        result.show(10, vertical=False, truncate=False)
        assert result.count() == 100
