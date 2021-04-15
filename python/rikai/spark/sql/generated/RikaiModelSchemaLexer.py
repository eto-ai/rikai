# Generated from src/main/antlr4/org/apache/spark/sql/ml/parser/RikaiModelSchema.g4 by ANTLR 4.7.2
from antlr4 import *
from io import StringIO
from typing.io import TextIO
import sys


def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2\n")
        buf.write("\u00a6\b\1\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7")
        buf.write("\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\4\f\t\f\4\r\t\r")
        buf.write("\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22\4\23")
        buf.write("\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30")
        buf.write("\4\31\t\31\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36")
        buf.write("\t\36\4\37\t\37\4 \t \4!\t!\4\"\t\"\4#\t#\4$\t$\4%\t%")
        buf.write("\3\2\3\2\3\3\3\3\3\4\3\4\3\5\3\5\3\6\3\6\3\6\3\6\3\6\3")
        buf.write("\6\3\6\3\7\3\7\3\7\3\7\3\7\3\7\3\b\3\b\3\b\6\bd\n\b\r")
        buf.write("\b\16\be\3\t\3\t\3\n\3\n\3\13\6\13m\n\13\r\13\16\13n\3")
        buf.write("\13\3\13\3\f\3\f\3\r\3\r\3\16\3\16\3\17\3\17\3\20\3\20")
        buf.write("\3\21\3\21\3\22\3\22\3\23\3\23\3\24\3\24\3\25\3\25\3\26")
        buf.write("\3\26\3\27\3\27\3\30\3\30\3\31\3\31\3\32\3\32\3\33\3\33")
        buf.write("\3\34\3\34\3\35\3\35\3\36\3\36\3\37\3\37\3 \3 \3!\3!\3")
        buf.write("\"\3\"\3#\3#\3$\3$\3%\3%\2\2&\3\3\5\4\7\5\t\6\13\7\r\b")
        buf.write("\17\t\21\2\23\2\25\n\27\2\31\2\33\2\35\2\37\2!\2#\2%\2")
        buf.write("\'\2)\2+\2-\2/\2\61\2\63\2\65\2\67\29\2;\2=\2?\2A\2C\2")
        buf.write("E\2G\2I\2\3\2\37\3\2\62;\4\2C\\c|\5\2\13\f\17\17\"\"\4")
        buf.write("\2CCcc\4\2DDdd\4\2EEee\4\2FFff\4\2GGgg\4\2HHhh\4\2IIi")
        buf.write("i\4\2JJjj\4\2KKkk\4\2LLll\4\2MMmm\4\2NNnn\4\2OOoo\4\2")
        buf.write("PPpp\4\2QQqq\4\2RRrr\4\2SSss\4\2TTtt\4\2UUuu\4\2VVvv\4")
        buf.write("\2WWww\4\2XXxx\4\2YYyy\4\2ZZzz\4\2[[{{\4\2\\\\||\2\u008d")
        buf.write("\2\3\3\2\2\2\2\5\3\2\2\2\2\7\3\2\2\2\2\t\3\2\2\2\2\13")
        buf.write("\3\2\2\2\2\r\3\2\2\2\2\17\3\2\2\2\2\25\3\2\2\2\3K\3\2")
        buf.write("\2\2\5M\3\2\2\2\7O\3\2\2\2\tQ\3\2\2\2\13S\3\2\2\2\rZ\3")
        buf.write("\2\2\2\17c\3\2\2\2\21g\3\2\2\2\23i\3\2\2\2\25l\3\2\2\2")
        buf.write("\27r\3\2\2\2\31t\3\2\2\2\33v\3\2\2\2\35x\3\2\2\2\37z\3")
        buf.write("\2\2\2!|\3\2\2\2#~\3\2\2\2%\u0080\3\2\2\2\'\u0082\3\2")
        buf.write("\2\2)\u0084\3\2\2\2+\u0086\3\2\2\2-\u0088\3\2\2\2/\u008a")
        buf.write("\3\2\2\2\61\u008c\3\2\2\2\63\u008e\3\2\2\2\65\u0090\3")
        buf.write("\2\2\2\67\u0092\3\2\2\29\u0094\3\2\2\2;\u0096\3\2\2\2")
        buf.write("=\u0098\3\2\2\2?\u009a\3\2\2\2A\u009c\3\2\2\2C\u009e\3")
        buf.write("\2\2\2E\u00a0\3\2\2\2G\u00a2\3\2\2\2I\u00a4\3\2\2\2KL")
        buf.write("\7>\2\2L\4\3\2\2\2MN\7.\2\2N\6\3\2\2\2OP\7@\2\2P\b\3\2")
        buf.write("\2\2QR\7<\2\2R\n\3\2\2\2ST\5;\36\2TU\5=\37\2UV\59\35\2")
        buf.write("VW\5? \2WX\5\33\16\2XY\5=\37\2Y\f\3\2\2\2Z[\5\27\f\2[")
        buf.write("\\\59\35\2\\]\59\35\2]^\5\27\f\2^_\5G$\2_\16\3\2\2\2`")
        buf.write("d\5\23\n\2ad\5\21\t\2bd\7a\2\2c`\3\2\2\2ca\3\2\2\2cb\3")
        buf.write("\2\2\2de\3\2\2\2ec\3\2\2\2ef\3\2\2\2f\20\3\2\2\2gh\t\2")
        buf.write("\2\2h\22\3\2\2\2ij\t\3\2\2j\24\3\2\2\2km\t\4\2\2lk\3\2")
        buf.write("\2\2mn\3\2\2\2nl\3\2\2\2no\3\2\2\2op\3\2\2\2pq\b\13\2")
        buf.write("\2q\26\3\2\2\2rs\t\5\2\2s\30\3\2\2\2tu\t\6\2\2u\32\3\2")
        buf.write("\2\2vw\t\7\2\2w\34\3\2\2\2xy\t\b\2\2y\36\3\2\2\2z{\t\t")
        buf.write("\2\2{ \3\2\2\2|}\t\n\2\2}\"\3\2\2\2~\177\t\13\2\2\177")
        buf.write("$\3\2\2\2\u0080\u0081\t\f\2\2\u0081&\3\2\2\2\u0082\u0083")
        buf.write("\t\r\2\2\u0083(\3\2\2\2\u0084\u0085\t\16\2\2\u0085*\3")
        buf.write("\2\2\2\u0086\u0087\t\17\2\2\u0087,\3\2\2\2\u0088\u0089")
        buf.write("\t\20\2\2\u0089.\3\2\2\2\u008a\u008b\t\21\2\2\u008b\60")
        buf.write("\3\2\2\2\u008c\u008d\t\22\2\2\u008d\62\3\2\2\2\u008e\u008f")
        buf.write("\t\23\2\2\u008f\64\3\2\2\2\u0090\u0091\t\24\2\2\u0091")
        buf.write("\66\3\2\2\2\u0092\u0093\t\25\2\2\u00938\3\2\2\2\u0094")
        buf.write("\u0095\t\26\2\2\u0095:\3\2\2\2\u0096\u0097\t\27\2\2\u0097")
        buf.write("<\3\2\2\2\u0098\u0099\t\30\2\2\u0099>\3\2\2\2\u009a\u009b")
        buf.write("\t\31\2\2\u009b@\3\2\2\2\u009c\u009d\t\32\2\2\u009dB\3")
        buf.write("\2\2\2\u009e\u009f\t\33\2\2\u009fD\3\2\2\2\u00a0\u00a1")
        buf.write("\t\34\2\2\u00a1F\3\2\2\2\u00a2\u00a3\t\35\2\2\u00a3H\3")
        buf.write("\2\2\2\u00a4\u00a5\t\36\2\2\u00a5J\3\2\2\2\6\2cen\3\2")
        buf.write("\3\2")
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
            "'<'", "','", "'>'", "':'" ]

    symbolicNames = [ "<INVALID>",
            "STRUCT", "ARRAY", "IDENTIFIER", "WS" ]

    ruleNames = [ "T__0", "T__1", "T__2", "T__3", "STRUCT", "ARRAY", "IDENTIFIER", 
                  "DIGIT", "LETTER", "WS", "A", "B", "C", "D", "E", "F", 
                  "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", 
                  "R", "S", "T", "U", "V", "W", "X", "Y", "Z" ]

    grammarFileName = "RikaiModelSchema.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.7.2")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


