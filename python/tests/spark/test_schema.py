from pyspark.sql.types import StructType, StructField, IntegerType, StringType

from rikai.spark.sql.schema import parse_schema


def test_parse_schema():
    struct = parse_schema("struct<foo:int, bar:string>")
    assert struct == StructType(
        [StructField("foo", IntegerType()), StructField("bar", StringType())]
    )
