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
import pyarrow as pa
import pyarrow.parquet as pq
from rikai.mixin import ToDict
from rikai.parquet.dataset import Dataset as RikaiDataset
import pandas as pd

__all__ = ["df_to_rikai"]


def df_to_rikai(
    df: pd.DataFrame, dest_path: str, schema: "pyspark.sql.types.StructType"
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
    """
    schema_json_str = schema.json()
    converted_df = conv(df, json.loads(schema_json_str))  # convert Rikai types
    # The spark schema is required so Spark can read it successfully
    table = pa.Table.from_pandas(converted_df).replace_schema_metadata(
        {RikaiDataset.SPARK_PARQUET_ROW_METADATA: schema_json_str}
    )
    pq.write_to_dataset(table, dest_path, use_legacy_dataset=False)


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
            converted[name] = ser.apply(
                lambda arr: [conv_elm(x, typ["elementType"]) for x in arr]
            )
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
        return udt.serialize(v)
