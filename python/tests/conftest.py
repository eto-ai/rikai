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

import os
import random
import string
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

# Third Party
import pytest
import torch
import torchvision
from pyspark.sql import SparkSession
from torch.utils.data import DataLoader  # Prevent DataLoader hangs

# Rikai
from rikai.spark.sql import init
from rikai.spark.utils import init_spark_session

random.seed(time.time())


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
                (
                    "spark.hadoop.fs.gs.impl",
                    "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem",
                ),
                (
                    "spark.hadoop.google.cloud.auth.service.account.enable",
                    "true",
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


@pytest.fixture
def gcs_tmpdir() -> str:
    """Create a temporary Google Cloud Storage (GCS) directory to test dataset.

    To enable GCS test, it requires both the AWS credentials,
    as well as `RIKAI_TEST_GCS_BUCKET` being set.
    """

    bucket = os.environ.get("RIKAI_TEST_GCS_BUCKET", None)
    if bucket is None:
        pytest.skip("skip test. RIKAI_TEST_GCS_BUCKET is not set")

    try:
        import gcsfs

        parsed = urlparse(bucket)

        fs = gcsfs.GCSFileSystem()
        try:
            fs.ls(parsed.netloc)
        except gcsfs.retry.HttpError as he:
            if he.code == 401:
                pytest.skip(
                    "skip test. Google Cloud Credentials are not set up."
                )
            else:
                raise
    except ImportError:
        pytest.skip("rikai[gcp] is not installed.")

    temp_dir = (
        bucket
        + "/"
        + "".join(random.choices(string.ascii_letters + string.digits, k=6))
    )
    return temp_dir
