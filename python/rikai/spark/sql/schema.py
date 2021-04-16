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

from antlr4 import CommonTokenStream, InputStream
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DataType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    StructField,
    StructType,
)

from rikai.spark.sql.generated.RikaiModelSchemaLexer import (
    RikaiModelSchemaLexer,
)
from rikai.spark.sql.generated.RikaiModelSchemaParser import (
    RikaiModelSchemaParser,
)
from rikai.spark.sql.generated.RikaiModelSchemaVisitor import (
    RikaiModelSchemaVisitor,
)

_SPARK_TYPE_MAPPING = {
    "bool": BooleanType(),
    "boolean": BooleanType(),
    "byte": ByteType(),
    "tinyint": ByteType(),
    "short": ShortType(),
    "smallint": ShortType(),
    "int": IntegerType(),
    "long": LongType(),
    "bigint": LongType(),
    "float": FloatType(),
    "double": DoubleType(),
    "str": StringType(),
    "string": StringType(),
    "binary": BinaryType(),
}


class SchemaError(Exception):
    def __init__(self, message: str):
        self.message = message


class SchemaBuilder(RikaiModelSchemaVisitor):
    def visitStructType(
        self, ctx: RikaiModelSchemaParser.StructTypeContext
    ) -> StructType:
        return StructType(
            [self.visitStructField(field) for field in ctx.field()]
        )

    def visitStructField(
        self, ctx: RikaiModelSchemaParser.StructFieldContext
    ) -> StructField:
        name = self.visit(ctx.identifier())
        dataType = self.visit(ctx.fieldType())
        return StructField(name, dataType)

    def visitArrayType(
        self, ctx: RikaiModelSchemaParser.ArrayTypeContext
    ) -> ArrayType:
        return ArrayType(self.visit(ctx.fieldType()))

    def visitUnquotedIdentifier(
        self, ctx: RikaiModelSchemaParser.UnquotedIdentifierContext
    ) -> str:
        identifer = ctx.IDENTIFIER().getText()
        if identifer[0].isnumeric():
            raise SchemaError(
                f'Identifier can not start with a digit: "{identifer}"'
            )
        return identifer

    def visitPlainFieldType(
        self, ctx: RikaiModelSchemaParser.PlainFieldTypeContext
    ) -> DataType:
        name = self.visit(ctx.identifier())
        try:
            return _SPARK_TYPE_MAPPING[name]
        except KeyError as e:
            # TODO: Support customized UDT
            raise SchemaError(f'Can not recognize type: "{name}"') from e


def parse_schema(schema_str: str) -> DataType:
    # input_stream = InputStream(schema_str)
    upper = CaseChangingStream(InputStream(schema_str), True)
    lexer = RikaiModelSchemaLexer(upper)
    stream = CommonTokenStream(lexer)
    parser = RikaiModelSchemaParser(stream)

    visitor = SchemaBuilder()
    schema = visitor.visit(parser.schema())
    # TODO(GH#112) we should add error listener to Antlr Parser.
    if schema is None:
        raise SchemaError(f"Invalid schema: '{schema_str}'")
    return schema


class CaseChangingStream:
    def __init__(self, stream, upper=False):
        self._stream = stream
        self._upper = upper

    def __getattr__(self, name):
        return self._stream.__getattribute__(name)

    def LA(self, offset):
        c = self._stream.LA(offset)
        if c <= 0:
            return c
        return ord(chr(c).upper() if self._upper else chr(c).lower())
