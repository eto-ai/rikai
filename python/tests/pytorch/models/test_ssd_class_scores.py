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
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from torchvision.models.detection.ssd import ssd300_vgg16
from torchvision.transforms import ToTensor

import rikai
from rikai.pytorch.models.ssd_class_scores import SSDClassScoresExtractor
from rikai.spark.types import Box2dType

model = ssd300_vgg16(pretrained=True)
model.eval()
class_scores_extractor = SSDClassScoresExtractor(model)
class_scores_extractor.eval()


def test_predict_value_equal(two_flickr_images: list):
    batch = [ToTensor()(two_flickr_images[0].to_pil())]
    with torch.no_grad():
        detections = model(batch)[0]
        class_scores = class_scores_extractor(batch)[0]

    # In torchvision 0.11.0, there is a bug in the order of finding max value
    # of a label.
    correct_idx = detections["scores"] == class_scores["scores"][:, 0]
    assert len(correct_idx) > len(detections["scores"]) * 0.8

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
    with mlflow.start_run():
        mlflow.pytorch.log_model(
            class_scores_extractor, "model", registered_model_name="classes"
        )

    m = mlflow.pytorch.load_model(f"models:/classes/1")
    assert_model_equal(m, class_scores_extractor)


def test_ssd_class_scores_module_with_spark(
    spark: SparkSession, two_flickr_rows: list
):
    with mlflow.start_run():
        rikai.mlflow.pytorch.log_model(
            model,
            "models",
            model_type="ssd_class_scores",
            registered_model_name="ssd_class_scores",
            labels={"func": "rikai.pytorch.models.torch.detection_label_fn"},
        )

    spark.sql("CREATE MODEL class_scores USING 'mlflow:/ssd_class_scores'")
    spark.sql("SHOW MODELS").show()

    spark.createDataFrame([two_flickr_rows[0]]).createOrReplaceTempView(
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
                            StructField("labels", ArrayType(StringType())),
                        ]
                    )
                ),
            )
        ]
    )

    assert df.count() == 1
    assert df.selectExpr("explode(confidence)").count() > 1
    dd = df.selectExpr("explode(confidence)").collect()[0]["col"]
    assert dd["label_ids"] == [1, 31]
    assert dd["labels"] == ["person", "handbag"]
