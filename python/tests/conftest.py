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

import logging
import os
import random
import string
import uuid
from pathlib import Path
from urllib.parse import urlparse

# Third Party
import pytest
import torchvision
from pyspark.sql import SparkSession

import torch

# Rikai
from rikai.spark.utils import get_default_jar_version, init_spark_session


@pytest.fixture(scope="session")
def spark(mlflow_tracking_uri: str) -> SparkSession:
    rikai_version = get_default_jar_version(use_snapshot=True)
    hadoop_version = "3.2.0"  # TODO(lei): get hadoop version

    return init_spark_session(
        dict(
            [
                (
                    "spark.jars.packages",
                    ",".join(
                        [
                            f"org.apache.hadoop:hadoop-aws:{hadoop_version}",
                            "ai.eto:rikai_2.12:{}".format(rikai_version),
                        ]
                    ),
                ),
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
                ("com.amazonaws.services.s3.enableV4", "true"),
                (
                    "fs.s3a.aws.credentials.provider",
                    "com.amazonaws.auth.InstanceProfileCredentialsProvider,"
                    "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
                ),
                (
                    "fs.AbstractFileSystem.s3a.impl",
                    "org.apache.hadoop.fs.s3a.S3A",
                ),
                ("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"),
                ("spark.hadoop.fs.s3a.access.key", os.environ.get("AWS")),
                (
                    "rikai.sql.ml.registry.mlflow.tracking_uri",
                    mlflow_tracking_uri,
                ),
            ]
        )
    )


@pytest.fixture
def asset_path() -> Path:
    return Path(__file__).parent / "assets"


@pytest.fixture
def s3_tmpdir() -> str:
    """Create a temporary S3 directory to test dataset.

    To enable s3 test, it requires both the AWS credentials,
    as well as `RIKAI_TEST_S3_URL` being set.
    """
    baseurl = os.environ.get("RIKAI_TEST_S3_URL", None)
    if baseurl is None:
        pytest.skip("Skipping test. RIKAI_TEST_S3_URL is not set")
    parsed = urlparse(baseurl)
    if parsed.scheme != "s3":
        raise ValueError("RIKAI_TEST_S3_URL must be a valid s3:// URL.")

    try:
        import boto3
        import botocore

        sts = boto3.client("sts")
        try:
            sts.get_caller_identity()
        except botocore.exceptions.ClientError:
            pytest.skip("AWS client can not be authenticated.")
    except ImportError:
        pytest.skip("Skip test, rikai[aws] (boto3) is not installed")

    temp_dir = (
        baseurl
        + "/"
        + "".join(random.choices(string.ascii_letters + string.digits, k=6))
    )
    yield temp_dir

    from pyarrow.fs import S3FileSystem

    s3fs = S3FileSystem()
    s3fs_path = urlparse(temp_dir)._replace(scheme="").geturl()
    try:
        s3fs.rm(s3fs_path, recursive=True)
    except Exception:
        logging.warn("Could not delete directory: %s", s3fs_path)


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

    To enable GCS test, it requires both the GCS credentials,
    as well as `RIKAI_TEST_GCS_URL` being set.

    Examples
    --------

    .. code-block:: bash

        $ export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
        $ export RIKAI_TEST_GCS_URL=gs://bucket
        $ pytest python/tests

    References
    ----------
    https://cloud.google.com/dataproc/docs/concepts/connectors/cloud-storage
    https://cloud.google.com/dataproc/docs/concepts/iam/iam
    """

    base_url = os.environ.get("RIKAI_TEST_GCS_URL", None)
    if base_url is None:
        pytest.skip("Skipping test. RIKAI_TEST_GCS_URL is not set")
    parsed = urlparse(base_url)
    if parsed.scheme != "gs":
        raise ValueError("RIKAI_TEST_GCS_URL must be a valid gs:// URL")

    fs = None
    try:
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        try:
            fs.ls(parsed.netloc)
        except gcsfs.retry.HttpError as he:
            if he.code == 401:
                pytest.skip(
                    "Skipping test. Google Cloud Credentials are not set up."
                )
            else:
                raise
    except ImportError:
        pytest.skip("rikai[gcp] is not installed.")

    temp_dir = (
        base_url
        + "/"
        + "".join(random.choices(string.ascii_letters + string.digits, k=6))
    )
    yield temp_dir

    assert fs is not None, "gcsfs must be initialized by now."
    parsed = urlparse(temp_dir)
    gcsfs_path = parsed._replace(scheme="").geturl()  # Erase scheme
    try:
        # Best effort to clean temp data
        fs.rm(gcsfs_path, recursive=True)
    except Exception:
        logging.error("Could not delete directory: %s", gcsfs_path)
