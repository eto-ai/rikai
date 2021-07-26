# Generated from src/main/antlr4/org/apache/spark/sql/ml/parser/RikaiModelSchema.g4 by ANTLR 4.8
from antlr4 import *
from io import StringIO
from typing.io import TextIO
import sys



def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2\n")
        buf.write(">\b\1\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7")
        buf.write("\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\3\2\3\2\3\3\3\3\3\4")
        buf.write("\3\4\3\5\3\5\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\7\3\7\3\7\3")
        buf.write("\7\3\7\3\7\3\b\3\b\3\b\6\b\60\n\b\r\b\16\b\61\3\t\3\t")
        buf.write("\3\n\3\n\3\13\6\139\n\13\r\13\16\13:\3\13\3\13\2\2\f\3")
        buf.write("\3\5\4\7\5\t\6\13\7\r\b\17\t\21\2\23\2\25\n\3\2\5\3\2")
        buf.write("\62;\4\2C\\c|\5\2\13\f\17\17\"\"\2?\2\3\3\2\2\2\2\5\3")
        buf.write("\2\2\2\2\7\3\2\2\2\2\t\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2")
        buf.write("\2\2\17\3\2\2\2\2\25\3\2\2\2\3\27\3\2\2\2\5\31\3\2\2\2")
        buf.write("\7\33\3\2\2\2\t\35\3\2\2\2\13\37\3\2\2\2\r&\3\2\2\2\17")
        buf.write("/\3\2\2\2\21\63\3\2\2\2\23\65\3\2\2\2\258\3\2\2\2\27\30")
        buf.write("\7>\2\2\30\4\3\2\2\2\31\32\7.\2\2\32\6\3\2\2\2\33\34\7")
        buf.write("@\2\2\34\b\3\2\2\2\35\36\7<\2\2\36\n\3\2\2\2\37 \7U\2")
        buf.write("\2 !\7V\2\2!\"\7T\2\2\"#\7W\2\2#$\7E\2\2$%\7V\2\2%\f\3")
        buf.write("\2\2\2&\'\7C\2\2\'(\7T\2\2()\7T\2\2)*\7C\2\2*+\7[\2\2")
        buf.write("+\16\3\2\2\2,\60\5\23\n\2-\60\5\21\t\2.\60\7a\2\2/,\3")
        buf.write("\2\2\2/-\3\2\2\2/.\3\2\2\2\60\61\3\2\2\2\61/\3\2\2\2\61")
        buf.write("\62\3\2\2\2\62\20\3\2\2\2\63\64\t\2\2\2\64\22\3\2\2\2")
        buf.write("\65\66\t\3\2\2\66\24\3\2\2\2\679\t\4\2\28\67\3\2\2\29")
        buf.write(":\3\2\2\2:8\3\2\2\2:;\3\2\2\2;<\3\2\2\2<=\b\13\2\2=\26")
        buf.write("\3\2\2\2\6\2/\61:\3\2\3\2")
        return buf.getvalue()


class RikaiModelSchemaLexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    T__0 = 1
    T__1 = 2
    T__2 = 3
    T__3 = 4
    STRUCT = 5
    ARRAY = 6
    IDENTIFIER = 7
    WS = 8

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE" ]

    literalNames = [ "<INVALID>",
            "'<'", "','", "'>'", "':'", "'STRUCT'", "'ARRAY'" ]

    symbolicNames = [ "<INVALID>",
            "STRUCT", "ARRAY", "IDENTIFIER", "WS" ]

    ruleNames = [ "T__0", "T__1", "T__2", "T__3", "STRUCT", "ARRAY", "IDENTIFIER", 
                  "DIGIT", "LETTER", "WS" ]

    grammarFileName = "RikaiModelSchema.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.8")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


