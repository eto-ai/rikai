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

import uuid
from pathlib import Path

# Third Party
import pytest
import torch
import torchvision
from pyspark.sql import SparkSession
from torch.utils.data import DataLoader  # Prevent DataLoader hangs

# Rikai
from rikai.spark.sql import init
from rikai.spark.utils import init_spark_session


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    return init_spark_session(
        dict(
            [
                ("spark.port.maxRetries", 128),
                (
                    "rikai.sql.ml.registry.test.impl",
                    "ai.eto.rikai.sql.model.testing.TestRegistry",
                ),
            ]
        )
    )


@pytest.fixture
def asset_path() -> Path:
    return Path(__file__).parent / "assets"


@pytest.fixture(scope="session")
def resnet_model_uri(tmp_path_factory):
    # Prepare model
    tmp_path = tmp_path_factory.mktemp(str(uuid.uuid4()))
    resnet = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
        progress=False,
    )
    model_uri = tmp_path / "resnet.pth"
    torch.save(resnet, model_uri)
    return model_uri
