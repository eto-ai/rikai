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

from rikai.spark.sql.callback_service import CallbackService
from rikai.spark.sql.codegen.fs import Registry


def test_cb_service_find_registry():
    cb = CallbackService(None)

    cb.resolve("rikai.spark.sql.codegen.fs.Registry", "s3://foo", "foo", {})
    assert isinstance(
        cb.registry_map["rikai.spark.sql.codegen.fs.Registry"], Registry
    )

    with pytest.raises(ModuleNotFoundError):
        cb.resolve("rikai.spark.not.exist.Registry", "s3://foo", "bar", {})
    assert len(cb.registry_map) == 1

