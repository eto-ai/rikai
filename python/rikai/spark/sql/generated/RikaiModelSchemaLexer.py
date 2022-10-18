# Generated from src/main/antlr4/org/apache/spark/sql/ml/parser/RikaiModelSchema.g4 by ANTLR 4.11.1
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
    from typing import TextIO
else:
    from typing.io import TextIO


def serializedATN():
    return [
        4,0,8,60,6,-1,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,
        6,7,6,2,7,7,7,2,8,7,8,2,9,7,9,1,0,1,0,1,1,1,1,1,2,1,2,1,3,1,3,1,
        4,1,4,1,4,1,4,1,4,1,4,1,4,1,5,1,5,1,5,1,5,1,5,1,5,1,6,1,6,1,6,4,
        6,46,8,6,11,6,12,6,47,1,7,1,7,1,8,1,8,1,9,4,9,55,8,9,11,9,12,9,56,
        1,9,1,9,0,0,10,1,1,3,2,5,3,7,4,9,5,11,6,13,7,15,0,17,0,19,8,1,0,
        3,1,0,48,57,2,0,65,90,97,122,3,0,9,10,13,13,32,32,61,0,1,1,0,0,0,
        0,3,1,0,0,0,0,5,1,0,0,0,0,7,1,0,0,0,0,9,1,0,0,0,0,11,1,0,0,0,0,13,
        1,0,0,0,0,19,1,0,0,0,1,21,1,0,0,0,3,23,1,0,0,0,5,25,1,0,0,0,7,27,
        1,0,0,0,9,29,1,0,0,0,11,36,1,0,0,0,13,45,1,0,0,0,15,49,1,0,0,0,17,
        51,1,0,0,0,19,54,1,0,0,0,21,22,5,60,0,0,22,2,1,0,0,0,23,24,5,44,
        0,0,24,4,1,0,0,0,25,26,5,62,0,0,26,6,1,0,0,0,27,28,5,58,0,0,28,8,
        1,0,0,0,29,30,5,83,0,0,30,31,5,84,0,0,31,32,5,82,0,0,32,33,5,85,
        0,0,33,34,5,67,0,0,34,35,5,84,0,0,35,10,1,0,0,0,36,37,5,65,0,0,37,
        38,5,82,0,0,38,39,5,82,0,0,39,40,5,65,0,0,40,41,5,89,0,0,41,12,1,
        0,0,0,42,46,3,17,8,0,43,46,3,15,7,0,44,46,5,95,0,0,45,42,1,0,0,0,
        45,43,1,0,0,0,45,44,1,0,0,0,46,47,1,0,0,0,47,45,1,0,0,0,47,48,1,
        0,0,0,48,14,1,0,0,0,49,50,7,0,0,0,50,16,1,0,0,0,51,52,7,1,0,0,52,
        18,1,0,0,0,53,55,7,2,0,0,54,53,1,0,0,0,55,56,1,0,0,0,56,54,1,0,0,
        0,56,57,1,0,0,0,57,58,1,0,0,0,58,59,6,9,0,0,59,20,1,0,0,0,4,0,45,
        47,56,1,0,1,0
    ]

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
        self.checkVersion("4.11.1")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


