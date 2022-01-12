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

from pathlib import Path

import mlflow
import numpy as np
import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import DoubleType, LongType, StructField, StructType
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

import rikai


def test_sklearn_linear_regression(tmp_path: Path, spark: SparkSession):
    # prepare training data
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    model = LinearRegression()

    tmp_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = "sqlite:///" + str(tmp_path / "tracking.db")
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run():
        model.fit(X, y)

        reg_model_name = "sklearn_linear_regression"
        model_name = "sk_lr_m"
        rikai.mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            schema="double",
            registered_model_name=reg_model_name,
        )

        spark.conf.set(
            "spark.rikai.sql.ml.registry.mlflow.tracking_uri", tracking_uri
        )
        spark.sql(
            f"""
            CREATE MODEL {model_name} USING 'mlflow:///{reg_model_name}';
            """
        )

        df = spark.range(2).selectExpr("id as x0", "id+1 as x1")
        df.createOrReplaceTempView("tbl_X")

        result = spark.sql(
            f"""
            select ML_PREDICT({model_name}, array(x0, x1)) as pred from tbl_X
            """
        )
        assert result.schema == StructType([StructField("pred", DoubleType())])
        assert result.count() == 2


def test_sklearn_random_forest(tmp_path: Path, spark: SparkSession):
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )

    # train a model
    model = RandomForestClassifier(max_depth=2, random_state=0)

    tmp_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = "sqlite:///" + str(tmp_path / "tracking.db")
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run():
        model.fit(X, y)

        reg_model_name = "sklearn_random_forest"
        model_name = "sk_rf_m"
        rikai.mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            schema="long",
            registered_model_name=reg_model_name,
        )

        spark.conf.set(
            "spark.rikai.sql.ml.registry.mlflow.tracking_uri", tracking_uri
        )
        spark.sql(
            f"""
            CREATE MODEL {model_name} USING 'mlflow:///{reg_model_name}';
            """
        )

        df = spark.range(2).selectExpr(
            "id as x0", "id+1 as x1", "id+2 as x2", "id+3 as x3"
        )
        df.createOrReplaceTempView("tbl_X")

        result = spark.sql(
            f"""
            select
                ML_PREDICT({model_name}, array(x0, x1, x2, x3)) as pred
            from tbl_X
            """
        )
        result.show()
        assert result.schema == StructType([StructField("pred", LongType())])
        assert (
            result.collect()
            == spark.createDataFrame([Row(pred=1), Row(pred=1)]).collect()
        )


def test_sklearn_pca(tmp_path: Path, spark: SparkSession):
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    model = PCA(n_components=2)

    tmp_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = "sqlite:///" + str(tmp_path / "tracking.db")
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run():
        model.fit(X)
        model_name = "sklearn_pca"
        reg_model_name = model_name
        rikai.mlflow.sklearn.log_model(
            model,
            "model",
            schema="array<float>",
            registered_model_name=reg_model_name,
        )
        spark.conf.set(
            "spark.rikai.sql.ml.registry.mlflow.tracking_uri", tracking_uri
        )
        spark.sql(
            f"""
            CREATE MODEL {model_name} USING 'mlflow:///{reg_model_name}';
            """
        )
        result = spark.sql(
            f"""
            select ML_PREDICT({model_name}, array(3, 2)) as pred
            """
        )
        result.show(1, vertical=False, truncate=False)
        assert (
            pytest.approx(result.head().pred) == model.transform([[3, 2]])[0]
        )
