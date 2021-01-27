#  Copyright 2020 Rikai Authors
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

from rikai.parquet.resolver import Resolver
from rikai.testing import assert_count_equal


def teardown_function(_):
    Resolver.reset()


def test_resolve_local_fs(tmp_path):
    for i in range(10):
        with (tmp_path / f"{i}.parquet").open(mode="w") as fobj:
            fobj.write("123")

    files = Resolver.resolve(tmp_path)
    expected_files = [
        "file://" + str(tmp_path / f"{i}.parquet") for i in range(10)
    ]
    assert_count_equal(expected_files, files)


def test_resolve_empty_dir(tmp_path):
    assert [] == Resolver.resolve(tmp_path)
