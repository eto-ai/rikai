from pathlib import Path

import mlflow
import numpy as np
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import DoubleType, LongType, StructField, StructType
from sklearn.datasets import make_classification
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

        registered_model_name = "sklearn_linear_regression"
        model_name = "sk_lr_m"
        rikai.mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            schema="double",
            registered_model_name=registered_model_name,
        )

        spark.conf.set(
            "rikai.sql.ml.registry.mlflow.tracking_uri", tracking_uri
        )
        spark.sql(
            f"""
        CREATE MODEL {model_name} USING 'mlflow:///{registered_model_name}';
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

        registered_model_name = "sklearn_random_forest"
        model_name = "sk_rf_m"
        rikai.mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            schema="long",
            registered_model_name=registered_model_name,
        )

        spark.conf.set(
            "rikai.sql.ml.registry.mlflow.tracking_uri", tracking_uri
        )
        spark.sql(
            f"""
            CREATE MODEL {model_name} USING 'mlflow:///{registered_model_name}';
        """
        )

        df = spark.range(2).selectExpr(
            "id as x0", "id+1 as x1", "id+2 as x2", "id+3 as x3"
        )
        df.createOrReplaceTempView("tbl_X")

        result = spark.sql(
            f"""
            select ML_PREDICT({model_name}, array(x0, x1, x2, x3)) as pred from tbl_X
        """
        )
        result.show()
        assert result.schema == StructType([StructField("pred", LongType())])
        assert (
            result.collect()
            == spark.createDataFrame([Row(pred=1), Row(pred=1)]).collect()
        )
