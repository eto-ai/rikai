# Generated from src/main/antlr4/org/apache/spark/sql/ml/parser/RikaiModelSchema.g4 by ANTLR 4.11.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,8,43,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,1,0,1,0,
        1,1,1,1,1,2,1,2,1,2,1,2,1,2,5,2,22,8,2,10,2,12,2,25,9,2,1,2,1,2,
        1,3,1,3,1,3,1,3,1,3,1,4,1,4,1,4,3,4,37,8,4,1,5,1,5,1,5,1,5,1,5,0,
        0,6,0,2,4,6,8,10,0,0,39,0,12,1,0,0,0,2,14,1,0,0,0,4,16,1,0,0,0,6,
        28,1,0,0,0,8,36,1,0,0,0,10,38,1,0,0,0,12,13,3,8,4,0,13,1,1,0,0,0,
        14,15,5,7,0,0,15,3,1,0,0,0,16,17,5,5,0,0,17,18,5,1,0,0,18,23,3,10,
        5,0,19,20,5,2,0,0,20,22,3,10,5,0,21,19,1,0,0,0,22,25,1,0,0,0,23,
        21,1,0,0,0,23,24,1,0,0,0,24,26,1,0,0,0,25,23,1,0,0,0,26,27,5,3,0,
        0,27,5,1,0,0,0,28,29,5,6,0,0,29,30,5,1,0,0,30,31,3,8,4,0,31,32,5,
        3,0,0,32,7,1,0,0,0,33,37,3,4,2,0,34,37,3,6,3,0,35,37,3,2,1,0,36,
        33,1,0,0,0,36,34,1,0,0,0,36,35,1,0,0,0,37,9,1,0,0,0,38,39,3,2,1,
        0,39,40,5,4,0,0,40,41,3,8,4,0,41,11,1,0,0,0,2,23,36
    ]

class RikaiModelSchemaParser ( Parser ):

    grammarFileName = "RikaiModelSchema.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "'<'", "','", "'>'", "':'", "'STRUCT'", 
                     "'ARRAY'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "STRUCT", "ARRAY", "IDENTIFIER", "WS" ]

    RULE_schema = 0
    RULE_identifier = 1
    RULE_struct = 2
    RULE_array = 3
    RULE_fieldType = 4
    RULE_field = 5

    ruleNames =  [ "schema", "identifier", "struct", "array", "fieldType", 
                   "field" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    STRUCT=5
    ARRAY=6
    IDENTIFIER=7
    WS=8

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.11.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class SchemaContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def fieldType(self):
            return self.getTypedRuleContext(RikaiModelSchemaParser.FieldTypeContext,0)


        def getRuleIndex(self):
            return RikaiModelSchemaParser.RULE_schema

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSchema" ):
                return visitor.visitSchema(self)
            else:
                return visitor.visitChildren(self)




    def schema(self):

        localctx = RikaiModelSchemaParser.SchemaContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_schema)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 12
            self.fieldType()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class IdentifierContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return RikaiModelSchemaParser.RULE_identifier

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class UnquotedIdentifierContext(IdentifierContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RikaiModelSchemaParser.IdentifierContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def IDENTIFIER(self):
            return self.getToken(RikaiModelSchemaParser.IDENTIFIER, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitUnquotedIdentifier" ):
                return visitor.visitUnquotedIdentifier(self)
            else:
                return visitor.visitChildren(self)



    def identifier(self):

        localctx = RikaiModelSchemaParser.IdentifierContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_identifier)
        try:
            localctx = RikaiModelSchemaParser.UnquotedIdentifierContext(self, localctx)
            self.enterOuterAlt(localctx, 1)
            self.state = 14
            self.match(RikaiModelSchemaParser.IDENTIFIER)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class StructContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return RikaiModelSchemaParser.RULE_struct

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class StructTypeContext(StructContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RikaiModelSchemaParser.StructContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def STRUCT(self):
            return self.getToken(RikaiModelSchemaParser.STRUCT, 0)
        def field(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(RikaiModelSchemaParser.FieldContext)
            else:
                return self.getTypedRuleContext(RikaiModelSchemaParser.FieldContext,i)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitStructType" ):
                return visitor.visitStructType(self)
            else:
                return visitor.visitChildren(self)



    def struct(self):

        localctx = RikaiModelSchemaParser.StructContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_struct)
        self._la = 0 # Token type
        try:
            localctx = RikaiModelSchemaParser.StructTypeContext(self, localctx)
            self.enterOuterAlt(localctx, 1)
            self.state = 16
            self.match(RikaiModelSchemaParser.STRUCT)
            self.state = 17
            self.match(RikaiModelSchemaParser.T__0)
            self.state = 18
            self.field()
            self.state = 23
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==2:
                self.state = 19
                self.match(RikaiModelSchemaParser.T__1)
                self.state = 20
                self.field()
                self.state = 25
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 26
            self.match(RikaiModelSchemaParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ArrayContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return RikaiModelSchemaParser.RULE_array

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class ArrayTypeContext(ArrayContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RikaiModelSchemaParser.ArrayContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def ARRAY(self):
            return self.getToken(RikaiModelSchemaParser.ARRAY, 0)
        def fieldType(self):
            return self.getTypedRuleContext(RikaiModelSchemaParser.FieldTypeContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitArrayType" ):
                return visitor.visitArrayType(self)
            else:
                return visitor.visitChildren(self)



    def array(self):

        localctx = RikaiModelSchemaParser.ArrayContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_array)
        try:
            localctx = RikaiModelSchemaParser.ArrayTypeContext(self, localctx)
            self.enterOuterAlt(localctx, 1)
            self.state = 28
            self.match(RikaiModelSchemaParser.ARRAY)
            self.state = 29
            self.match(RikaiModelSchemaParser.T__0)
            self.state = 30
            self.fieldType()
            self.state = 31
            self.match(RikaiModelSchemaParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class FieldTypeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return RikaiModelSchemaParser.RULE_fieldType

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class NestedStructTypeContext(FieldTypeContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RikaiModelSchemaParser.FieldTypeContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def struct(self):
            return self.getTypedRuleContext(RikaiModelSchemaParser.StructContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitNestedStructType" ):
                return visitor.visitNestedStructType(self)
            else:
                return visitor.visitChildren(self)


    class PlainFieldTypeContext(FieldTypeContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RikaiModelSchemaParser.FieldTypeContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def identifier(self):
            return self.getTypedRuleContext(RikaiModelSchemaParser.IdentifierContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPlainFieldType" ):
                return visitor.visitPlainFieldType(self)
            else:
                return visitor.visitChildren(self)


    class NestedArrayTypeContext(FieldTypeContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RikaiModelSchemaParser.FieldTypeContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def array(self):
            return self.getTypedRuleContext(RikaiModelSchemaParser.ArrayContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitNestedArrayType" ):
                return visitor.visitNestedArrayType(self)
            else:
                return visitor.visitChildren(self)



    def fieldType(self):

        localctx = RikaiModelSchemaParser.FieldTypeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_fieldType)
        try:
            self.state = 36
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [5]:
                localctx = RikaiModelSchemaParser.NestedStructTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 33
                self.struct()
                pass
            elif token in [6]:
                localctx = RikaiModelSchemaParser.NestedArrayTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 34
                self.array()
                pass
            elif token in [7]:
                localctx = RikaiModelSchemaParser.PlainFieldTypeContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 35
                self.identifier()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class FieldContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return RikaiModelSchemaParser.RULE_field

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class StructFieldContext(FieldContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a RikaiModelSchemaParser.FieldContext
            super().__init__(parser)
            self.name = None # IdentifierContext
            self.copyFrom(ctx)

        def fieldType(self):
            return self.getTypedRuleContext(RikaiModelSchemaParser.FieldTypeContext,0)

        def identifier(self):
            return self.getTypedRuleContext(RikaiModelSchemaParser.IdentifierContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitStructField" ):
                return visitor.visitStructField(self)
            else:
                return visitor.visitChildren(self)



    def field(self):

        localctx = RikaiModelSchemaParser.FieldContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_field)
        try:
            localctx = RikaiModelSchemaParser.StructFieldContext(self, localctx)
            self.enterOuterAlt(localctx, 1)
            self.state = 38
            localctx.name = self.identifier()
            self.state = 39
            self.match(RikaiModelSchemaParser.T__3)
            self.state = 40
            self.fieldType()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





