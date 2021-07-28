# Generated from src/main/antlr4/org/apache/spark/sql/ml/parser/RikaiModelSchema.g4 by ANTLR 4.8
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .RikaiModelSchemaParser import RikaiModelSchemaParser
else:
    from RikaiModelSchemaParser import RikaiModelSchemaParser

# This class defines a complete generic visitor for a parse tree produced by RikaiModelSchemaParser.

class RikaiModelSchemaVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by RikaiModelSchemaParser#schema.
    def visitSchema(self, ctx:RikaiModelSchemaParser.SchemaContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RikaiModelSchemaParser#unquotedIdentifier.
    def visitUnquotedIdentifier(self, ctx:RikaiModelSchemaParser.UnquotedIdentifierContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RikaiModelSchemaParser#structType.
    def visitStructType(self, ctx:RikaiModelSchemaParser.StructTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RikaiModelSchemaParser#arrayType.
    def visitArrayType(self, ctx:RikaiModelSchemaParser.ArrayTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RikaiModelSchemaParser#nestedStructType.
    def visitNestedStructType(self, ctx:RikaiModelSchemaParser.NestedStructTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RikaiModelSchemaParser#nestedArrayType.
    def visitNestedArrayType(self, ctx:RikaiModelSchemaParser.NestedArrayTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RikaiModelSchemaParser#plainFieldType.
    def visitPlainFieldType(self, ctx:RikaiModelSchemaParser.PlainFieldTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by RikaiModelSchemaParser#structField.
    def visitStructField(self, ctx:RikaiModelSchemaParser.StructFieldContext):
        return self.visitChildren(ctx)



del RikaiModelSchemaParser