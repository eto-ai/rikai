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
import mlflow
import py4j
import pytest
import torch
from mlflow.tracking import MlflowClient
from pyspark.sql import Row, SparkSession
from utils import check_ml_predict

import rikai
from rikai.spark.sql.codegen.mlflow_registry import MlflowModelSpec
from rikai.spark.sql.schema import parse_schema


@pytest.fixture(scope="session")
def mlflow_client_with_tracking_uri(
    tmp_path_factory, resnet_model_uri: str
) -> (MlflowClient, str):
    tmp_path = tmp_path_factory.mktemp("mlflow")
    tmp_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = "sqlite:///" + str(tmp_path / "tracking.db")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment("rikai-test", str(tmp_path))
    # simpliest
    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_param("optimizer", "Adam")
        # Fake training loop
        model = torch.load(resnet_model_uri)
        artifact_path = "model"

        schema = (
            "STRUCT<boxes:ARRAY<ARRAY<float>>,"
            "scores:ARRAY<float>,label_ids:ARRAY<int>>"
        )
        pre_processing = (
            "rikai.contrib.torch.transforms."
            "fasterrcnn_resnet50_fpn.pre_processing"
        )
        post_processing = (
            "rikai.contrib.torch.transforms."
            "fasterrcnn_resnet50_fpn.post_processing"
        )
        rikai.mlflow.pytorch.log_model(
            model,  # same as vanilla mlflow
            artifact_path,  # same as vanilla mlflow
            schema,
            pre_processing,
            post_processing,
            registered_model_name="rikai-test",  # same as vanilla mlflow
        )

    # vanilla mlflow
    with mlflow.start_run():
        mlflow.pytorch.log_model(
            model, artifact_path, registered_model_name="vanilla-mlflow"
        )
        mlflow.set_tags(
            {
                "rikai.model.flavor": "pytorch",
                "rikai.output.schema": schema,
                "rikai.transforms.pre": pre_processing,
                "rikai.transforms.post": post_processing,
            }
        )

    # vanilla mlflow no tags
    with mlflow.start_run():
        mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name="vanilla-mlflow-no-tags",
        )

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
                "rikai.output.schema": schema,
                "rikai.transforms.pre": "wrong_pre",
                "rikai.transforms.post": "wrong_post",
            }
        )
    return mlflow.tracking.MlflowClient(tracking_uri), tracking_uri


@pytest.fixture(scope="session")
def mlflow_client(mlflow_client_with_tracking_uri):
    return mlflow_client_with_tracking_uri[0]


@pytest.fixture(scope="session")
def mlflow_tracking_uri(mlflow_client_with_tracking_uri):
    return mlflow_client_with_tracking_uri[1]


def test_modelspec(mlflow_client: MlflowClient):
    mv = mlflow_client.search_model_versions("name='rikai-test'")[0]
    run = mlflow_client.get_run(run_id=mv.run_id)
    spec = MlflowModelSpec(
        "models:/rikai-test/{}".format(mv.version),
        run.data.tags,
        tracking_uri="fake",
    )
    assert spec.flavor == "pytorch"
    assert spec.schema == parse_schema(
        "STRUCT<boxes:ARRAY<ARRAY<float>>,"
        "scores:ARRAY<float>, label_ids:ARRAY<int>>"
    )
    assert spec._spec["transforms"]["pre"] == (
        "rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn"
        ".pre_processing"
    )
    assert spec._spec["transforms"]["post"] == (
        "rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn."
        "post_processing"
    )
    assert spec.model_uri == "models:/rikai-test/{}".format(mv.version)


@pytest.mark.timeout(600)
def test_mlflow_model_from_model_version(
    spark: SparkSession, mlflow_client: MlflowClient
):
    # peg to a particular version of a model
    spark.sql("CREATE MODEL resnet_m_fizz USING 'mlflow:/rikai-test/1'")
    check_ml_predict(spark, "resnet_m_fizz")

    # use the latest version in a given stage (omitted means none)
    spark.sql("CREATE MODEL resnet_m_buzz USING 'mlflow:/rikai-test'")
    check_ml_predict(spark, "resnet_m_buzz")


@pytest.mark.timeout(600)
def test_mlflow_model_without_custom_logger(
    spark: SparkSession, mlflow_client: MlflowClient
):
    spark.sql("CREATE MODEL vanilla_ice USING 'mlflow:/vanilla-mlflow/1'")
    check_ml_predict(spark, "vanilla_ice")

    schema = (
        "STRUCT<boxes:ARRAY<ARRAY<float>>,"
        "scores:ARRAY<float>,label_ids:ARRAY<int>>"
    )
    pre_processing = (
        "rikai.contrib.torch.transforms."
        "fasterrcnn_resnet50_fpn.pre_processing"
    )
    post_processing = (
        "rikai.contrib.torch.transforms."
        "fasterrcnn_resnet50_fpn.post_processing"
    )
    spark.sql(
        (
            "CREATE MODEL vanilla_fire "
            "FLAVOR pytorch "
            "PREPROCESSOR '{}' "
            "POSTPROCESSOR '{}' "
            "RETURNS {} "
            "USING 'mlflow:/vanilla-mlflow-no-tags/1'"
        ).format(pre_processing, post_processing, schema)
    )
    check_ml_predict(spark, "vanilla_fire")

    spark.sql(
        (
            "CREATE MODEL vanilla_fixer "
            "FLAVOR pytorch "
            "PREPROCESSOR '{}' "
            "POSTPROCESSOR '{}' "
            "RETURNS {} "
            "USING 'mlflow:/vanilla-mlflow-wrong-tags/1'"
        ).format(pre_processing, post_processing, schema)
    )
    check_ml_predict(spark, "vanilla_fixer")


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
