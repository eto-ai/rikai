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
import re

from pyspark.sql import DataFrame

from rikai.__version__ import version
from .conf import CONF_PARQUET_BLOCK_SIZE, DEFAULT_ROW_GROUP_SIZE_BYTES


def df_to_rikai(
    df: DataFrame,
    uri: str,
    parquet_row_group_size_bytes: int = DEFAULT_ROW_GROUP_SIZE_BYTES,
):
    (
        df.write.format("rikai")
        .option(CONF_PARQUET_BLOCK_SIZE, parquet_row_group_size_bytes)
        .save(uri)
    )


def get_default_jar_version(use_snapshot=True):
    """
    Make it easier to reference the jar version in notebooks and conftest.

    Parameters
    ----------
    use_snapshot: bool, default True
        If True then map `*dev0` versions to `-SNAPSHOT`
    """
    pattern = re.compile(r"([\d]+.[\d]+.[\d]+)")
    match = re.search(pattern, version)
    if not match:
        raise ValueError("Ill-formed version string {}".format(version))
    match_str = match.group(1)
    if use_snapshot and (len(match_str) < len(version)):
        return match_str + "-SNAPSHOT"
    return match_str
