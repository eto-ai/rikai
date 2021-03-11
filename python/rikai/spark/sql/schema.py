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

from antlr4 import InputStream, CommonTokenStream
from pyspark.sql.types import DataType, StructType, StructField, StringType, IntegerType

from rikai.spark.sql.generated.RikaiModelSchemaLexer import (
    RikaiModelSchemaLexer,
)
from rikai.spark.sql.generated.RikaiModelSchemaParser import (
    RikaiModelSchemaParser,
)
from rikai.spark.sql.generated.RikaiModelSchemaVisitor import (
    RikaiModelSchemaVisitor,
)


class SchemaBuilder(RikaiModelSchemaVisitor):
    def visitStructType(
        self, ctx: RikaiModelSchemaParser.StructTypeContext
    ) -> StructType:
        print("Visit struct: ", ctx.STRUCT(), ctx.field())
        return StructType(
            [self.visitStructField(field) for field in ctx.field()]
        )

    def visitStructField(
        self, ctx: RikaiModelSchemaParser.StructFieldContext
    ) -> StructField:
        name = self.visit(ctx.identifier())
        dataType = self.visit(ctx.fieldType())
        return StructField(name, dataType)

    def visitUnquotedIdentifier(
        self, ctx: RikaiModelSchemaParser.UnquotedIdentifierContext
    ):
        return ctx.IDENTIFIER().getText()

    def visitPlainFieldType(
        self, ctx: RikaiModelSchemaParser.PlainFieldTypeContext
    ) -> DataType:
        name = self.visit(ctx.identifier()).lower()
        return {
            'int': IntegerType(),
            'str': StringType(),
            'string': StringType(),
        }[name]


def parse_schema(schema_str: str) -> DataType:
    input_stream = InputStream(schema_str)
    lexer = RikaiModelSchemaLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = RikaiModelSchemaParser(stream)

    visitor = SchemaBuilder()
    schema = visitor.visit(parser.schema())
    return schema
