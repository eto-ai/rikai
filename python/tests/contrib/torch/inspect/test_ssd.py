#  Copyright 2022 Rikai Authors
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

import mlflow
import torch
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (
    StructType,
    ArrayType,
    StructField,
    FloatType,
    IntegerType,
)
from torchvision.models.detection.ssd import ssd300_vgg16
from torchvision.transforms import ToTensor

import rikai
from rikai.contrib.torch.inspect.ssd import (
    SSDClassScoresExtractor,
    pre_processing,
)
from rikai.spark.types import Box2dType
from rikai.types import Image

TEST_IMAGE = Image(
    "http://farm2.staticflickr.com/1129/4726871278_4dd241a03a_z.jpg"
)

model = ssd300_vgg16(pretrained=True)
model.eval()
class_scores_extractor = SSDClassScoresExtractor(model)
class_scores_extractor.eval()


def test_predict_value_equal():
    batch = [ToTensor()(TEST_IMAGE.to_pil())]
    with torch.no_grad():
        detections = model(batch)[0]
        class_scores = class_scores_extractor(batch)[0]

    # In torchvision 0.11.0, there is a bug in the order to find max value
    # of a label.
    correct_idx = detections["scores"] == class_scores["scores"][:, 0]
    assert len(correct_idx) > len(detections["scores"]) * 0.8

    print("BUG FROM PYTORCH")
    print(
        f"SCORES: pytorch: {detections['scores'][~correct_idx]},"
        f" we got: {class_scores['scores'][~correct_idx]}"
    )
    print(
        f"LABELS: pytorch: {detections['labels'][~correct_idx]}, "
        f" we got: {class_scores['labels'][~correct_idx]}"
    )

    assert torch.equal(
        detections["boxes"][correct_idx], class_scores["boxes"][correct_idx]
    )
    assert torch.equal(
        detections["scores"][correct_idx],
        class_scores["scores"][correct_idx][:, 0],
    )
    assert torch.equal(
        detections["labels"][correct_idx],
        class_scores["labels"][correct_idx][:, 0],
    )


def assert_model_equal(expect: torch.nn.Module, actual: torch.nn.Module):
    for act, exp in zip(
        actual.parameters(recurse=True),
        expect.parameters(recurse=True),
    ):
        assert torch.equal(act, exp)


def test_ssd_class_score_module_serialization(tmp_path: Path):
    # test save model
    torch.save(class_scores_extractor, tmp_path / "model.pt")

    m = torch.load(tmp_path / "model.pt")
    assert_model_equal(class_scores_extractor, m)

    script_model = torch.jit.script(class_scores_extractor)
    torch.jit.save(script_model, tmp_path / "script.pt")
    actual_script_model = torch.jit.load(tmp_path / "script.pt")
    assert_model_equal(script_model, actual_script_model)


def test_ssd_class_score_module_mlflow(tmp_path: Path):
    tracking_uri = "sqlite:///" + str(tmp_path / "tracking.db")
    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run():
        mlflow.pytorch.log_model(
            class_scores_extractor, "model", registered_model_name="classes"
        )

    m = mlflow.pytorch.load_model(f"models:/classes/1")
    assert_model_equal(m, class_scores_extractor)


def test_ssd_class_scores_module_with_spark(spark: SparkSession):
    rikai.mlflow.pytorch.log_model(
        class_scores_extractor,
        "models",
        SSDClassScoresExtractor.SCHEMA,
        registered_model_name="ssd_class_scores",
        pre_processing="rikai.contrib.torch.inspect.ssd.class_scores_extractor_pre_processing",  # noqa: E501
        post_processing="rikai.contrib.torch.inspect.ssd.class_scores_extractor_post_processing",  # noqa: E501
    )

    spark.sql("CREATE MODEL class_scores USING 'mlflow:/ssd_class_scores'")
    spark.sql("SHOW MODELS").show()

    spark.createDataFrame([Row(image=TEST_IMAGE)]).createOrReplaceTempView(
        "images"
    )

    df = spark.sql(
        "SELECT ML_PREDICT(class_scores, image) as confidence FROM images"
    )
    df.cache()
    df.show()

    assert df.schema == StructType(
        [
            StructField(
                "confidence",
                ArrayType(
                    StructType(
                        [
                            StructField("box", Box2dType()),
                            StructField("scores", ArrayType(FloatType())),
                            StructField("label_ids", ArrayType(IntegerType())),
                        ]
                    )
                ),
            )
        ]
    )

    assert df.count() == 1
    assert df.selectExpr("explode(confidence)").count() > 1
