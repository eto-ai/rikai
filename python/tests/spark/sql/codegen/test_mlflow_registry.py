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

import py4j
import pytest
from mlflow.tracking import MlflowClient
import mlflow
from pyspark.sql import SparkSession
from utils import check_ml_predict
import torch

import rikai
from rikai.contrib.torch.detections import OUTPUT_SCHEMA
from rikai.spark.sql.codegen.mlflow_registry import MlflowModelSpec
from rikai.spark.sql.schema import parse_schema
from rikai.spark.sql.codegen.mlflow_registry import CONF_MLFLOW_TRACKING_URI


def test_modelspec(mlflow_client: MlflowClient):
    mv = mlflow_client.search_model_versions("name='rikai-test'")[0]
    run = mlflow_client.get_run(run_id=mv.run_id)
    spec = MlflowModelSpec(
        "models:/rikai-test/{}".format(mv.version),
        run.data.tags,
        tracking_uri="fake",
    )
    assert spec.flavor == "pytorch"
    assert spec.schema == parse_schema(OUTPUT_SCHEMA)
    assert spec.model_uri == "models:/rikai-test/{}".format(mv.version)


@pytest.mark.timeout(200)
def test_mlflow_model_from_model_version(
    spark: SparkSession, mlflow_client: MlflowClient, two_flickr_rows: list
):
    # peg to a particular version of a model
    spark.sql("CREATE MODEL resnet_m_fizz USING 'mlflow:/rikai-test/1'")
    check_ml_predict(spark, "resnet_m_fizz", two_flickr_rows)

    # use the latest version in a given stage (omitted means none)
    spark.sql("CREATE MODEL resnet_m_buzz USING 'mlflow:/rikai-test'")
    check_ml_predict(spark, "resnet_m_buzz", two_flickr_rows)


@pytest.mark.skip
@pytest.mark.timeout(200)
def test_mlflow_model_without_custom_logger(
    spark: SparkSession,
    mlflow_client: MlflowClient,
    two_flickr_rows: list,
    resnet_model_uri: str,
):
    artifact_path = "models"
    model = torch.load(resnet_model_uri)
    # vanilla mlflow no tags
    with mlflow.start_run():
        mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name="vanilla-mlflow-no-tags",
        )

    spark.sql("CREATE MODEL vanilla_ice USING 'mlflow:/vanilla-mlflow/1'")
    check_ml_predict(spark, "vanilla_ice", two_flickr_rows)

    schema = OUTPUT_SCHEMA
    spark.sql(
        "CREATE MODEL vanilla_fire "
        "FLAVOR pytorch "
        f"RETURNS {schema}"
        "USING 'mlflow:/vanilla-mlflow-no-tags/1'"
    )
    check_ml_predict(spark, "vanilla_fire", two_flickr_rows)

    # vanilla mlflow wrong tags
    with mlflow.start_run():
        mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name="vanilla-mlflow-wrong-tags",
        )
        mlflow.set_tags(
            {
                "rikai.model.flavor": "pytorch",
                "rikai.output.schema": OUTPUT_SCHEMA,
            }
        )

    spark.sql(
        (
            "CREATE MODEL vanilla_fixer "
            "FLAVOR pytorch "
            f"RETURNS {schema}"
            "USING 'mlflow:/vanilla-mlflow-wrong-tags/1'"
        )
    )
    check_ml_predict(spark, "vanilla_fixer", two_flickr_rows)


@pytest.mark.timeout(600)
def test_mlflow_model_error_handling(
    spark: SparkSession, mlflow_client: MlflowClient
):
    with pytest.raises(
        py4j.protocol.Py4JJavaError,
        match=r".*URI with 2 forward slashes is not supported.*",
    ):
        spark.sql("CREATE MODEL two_slash USING 'mlflow://vanilla-mlflow/1'")

    with pytest.raises(
        py4j.protocol.Py4JJavaError,
        match=r".*Model registry scheme 'wrong' is not supported.*",
    ):
        spark.sql("CREATE MODEL wrong_uri USING 'wrong://vanilla-mlflow/1'")


def test_mlflow_model_type(
    tmp_path: Path,
    resnet_model_uri: str,
    spark: SparkSession,
    two_flickr_rows: list,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = "sqlite:///" + str(tmp_path / "tracking.db")
    mlflow.set_tracking_uri(tracking_uri)
    name = "resnet50_fpn_model_type"
    with mlflow.start_run():
        model = torch.load(resnet_model_uri)
        rikai.mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            model_type="fasterrcnn_resnet50_fpn",
            registered_model_name=name,
        )
        spark.conf.set(CONF_MLFLOW_TRACKING_URI, tracking_uri)
        spark.sql(f"CREATE MODEL {name} USING 'mlflow:/{name}'")
        check_ml_predict(spark, name, two_flickr_rows)
