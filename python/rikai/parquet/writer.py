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

"""Tools to write Rikai format datasets to disk
"""

import json
from typing import Union

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from rikai.mixin import ToDict
from rikai.parquet.dataset import Dataset as RikaiDataset

__all__ = ["df_to_rikai"]


def df_to_rikai(
    df: pd.DataFrame,
    dest_path: str,
    schema: "pyspark.sql.types.StructType",
    partition_cols: Union[str, list] = None,
    max_rows_per_file: int = None,
    mode: str = None,
):
    """Write the given pandas dataframe to the given location using the
    specified (pyspark) schema

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to write to Rikai format
    dest_path: str
        The destination path to write df to
    schema: StructType
        The pyspark schema required to convert Rikai types and make the
        resulting rikai dataset readable by spark
    partition_cols: str or list, default None
        Columns to partition on
    max_rows_per_file: int, default None
        passed to Arrow Dataset.write_dataset
    mode: str, default None
        This controls `existing_data_behavior` in pyarrow.
        {'error', 'overwrite_or_ignore', 'delete_matching'}
        None will default to Pyarrow's current default ('error')
        https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html
    """
    schema_json_str = schema.json()
    converted_df = conv(df, json.loads(schema_json_str))  # convert Rikai types
    # The spark schema is required so Spark can read it successfully
    table = pa.Table.from_pandas(converted_df).replace_schema_metadata(
        {RikaiDataset.SPARK_PARQUET_ROW_METADATA: schema_json_str}
    )
    kwds = {}
    if partition_cols is not None:
        if isinstance(partition_cols, str):
            partition_cols = [partition_cols]
        kwds["partitioning"] = partition_cols
        kwds["partitioning_flavor"] = "hive"
    if max_rows_per_file is not None:
        kwds["max_rows_per_file"] = max_rows_per_file
        if max_rows_per_file < 1024 * 1024:  # default max_rows_per_group
            # set max_rows_per_group to be equal or smaller
            kwds["max_rows_per_group"] = max_rows_per_file
    if mode is not None:
        kwds["existing_data_behavior"] = mode
    ds.write_dataset(table, dest_path, format="parquet", **kwds)


def conv(df: pd.DataFrame, schema: dict):
    """Convert the given pandas dataframe to native types

    Returns
    -------
    converted: pd.DataFrame
        A dataframe where all Rikai types have been converted
    """
    assert schema["type"] == "struct"
    converted = {}
    for field in schema["fields"]:
        name = field["name"]
        ser = df[name]
        typ = field["type"]
        if not isinstance(typ, dict):
            converted[name] = ser
        elif is_udt_type(typ):
            converted[name] = ser.apply(lambda x: _conv_udt(x, typ))
        elif typ["type"] == "array":
            elm_type = typ["elementType"]
            if isinstance(elm_type, dict):
                converted[name] = ser.apply(
                    lambda arr: [conv_elm(x, typ["elementType"]) for x in arr]
                )
            else:
                converted[name] = ser
        elif typ["type"] == "struct":
            converted[name] = ser.apply(lambda d: conv_elm(d, typ))
        else:
            converted[name] = ser
    return pd.DataFrame(converted)


def is_udt_type(typ: dict):
    """Check whether a given field type information is a udt"""
    if typ["type"] == "udt":
        udt = RikaiDataset._find_udt(typ["pyClass"])
        return udt is not None
    return False


def conv_elm(elm, schema):
    """Convert a single value given it's schema information"""
    if is_udt_type(schema):
        return elm.to_dict()
    elif schema["type"] == "array":
        return [conv_elm(x, schema["elementType"]) for x in elm]
    elif schema["type"] == "struct":
        converted = {}
        for field in schema["fields"]:
            name = field["name"]
            typ = field["type"]
            if not isinstance(typ, dict):
                converted[name] = elm[name]
            elif is_udt_type(typ):
                converted[name] = _conv_udt(elm[name], typ)
            elif typ["type"] == "array":
                converted[name] = [
                    conv_elm(x, typ["elementType"]) for x in elm[name]
                ]
            elif typ["type"] == "struct":
                converted[name] = conv_elm(elm[name], typ)
            else:
                converted[name] = elm[name]
        return converted
    else:
        return elm


def _conv_udt(v, typ):
    """Convert the UDT value to a serializable format"""
    if isinstance(v, ToDict):
        return v.to_dict()
    else:
        assert typ["type"] == "udt"
        udt = RikaiDataset._find_udt(typ["pyClass"])
        data = udt.serialize(v)
        return data
