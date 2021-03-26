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

from pyspark.sql import DataFrame

DEFAULT_ROW_GROUP_SIZE_BYTES = 32 * 1024 * 1024

def df_to_parquet(df: DataFrame, uri: str,
        parquet_row_group_size_bytes: int=DEFAULT_ROW_GROUP_SIZE_BYTES):
    (df.write.option("parquet.block.size", parquet_row_group_size_bytes)
    .parquet(uri))

