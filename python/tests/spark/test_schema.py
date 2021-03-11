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
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    StructField,
    StructType,
)

from rikai.spark.sql.schema import SchemaError, parse_schema


def test_parse_schema():
    struct = parse_schema("struct<foo:int, bar:string>")
    assert struct == StructType(
        [StructField("foo", IntegerType()), StructField("bar", StringType())]
    )


def test_primitives():
    assert BooleanType() == parse_schema("bool")
    assert BooleanType() == parse_schema("boolean")

    assert ByteType() == parse_schema("byte")
    assert ByteType() == parse_schema("tinyint")

    assert ShortType() == parse_schema("short")
    assert ShortType() == parse_schema("smallint")

    assert IntegerType() == parse_schema("int")
    assert FloatType() == parse_schema("float")
    assert DoubleType() == parse_schema("double")

    assert StringType() == parse_schema("string")
    assert BinaryType() == parse_schema("binary")


def test_nested_array():
    schema = StructType(
        [
            StructField("id", IntegerType()),
            StructField("scores", ArrayType(LongType())),
        ]
    )

    assert schema == parse_schema(schema.simpleString())
    assert schema == parse_schema("struct<id:int, scores:array<bigint>>")
    assert schema == parse_schema("struct<id:int, scores:array<long>>")


def test_invalid_identifier():
    with pytest.raises(SchemaError, match=r".*can not start with a digit.*"):
        parse_schema("struct<0id:int>")
    with pytest.raises(SchemaError, match=r".*can not start with a digit.*"):
        parse_schema("struct<id:8float>")
