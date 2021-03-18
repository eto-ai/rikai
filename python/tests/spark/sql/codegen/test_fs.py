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

import pytest

from rikai.spark.sql.codegen.fs import ModelSpec
from rikai.spark.sql.exceptions import SpecError


def test_validate_yaml_spec():
    ModelSpec(
        {
            "version": "1.2",
            "name": "test_yaml_model",
            "schema": "long",
            "model": {
                "uri": "s3://bucket/to/model.pt",
                "unspecified_field": True,
            },
            "options": {
                "gpu": "true",
                "batch_size": 123,
            },
        },
    )


def test_validate_misformed_spec():
    with pytest.raises(SpecError):
        ModelSpec({})

    with pytest.raises(SpecError, match=".*version' is a required property.*"):
        ModelSpec(
            {
                "name": "test_yaml_model",
                "schema": "long",
                "model": {"uri": "s3://foo/bar"},
            },
        )

    with pytest.raises(SpecError, match=".*'model' is a required property.*"):
        ModelSpec(
            {
                "version": "1.0",
                "name": "test_yaml_model",
                "schema": "long",
            },
        )

    with pytest.raises(SpecError, match=".*'uri' is a required property.*"):
        ModelSpec(
            {
                "version": "1.0",
                "name": "test_yaml_model",
                "schema": "long",
                "model": {},
            },
        )
