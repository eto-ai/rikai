/*
 * Copyright 2021 Rikai authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

// Grammar for Rikai-extensions.
grammar RikaiExtSqlBase;

singleStatement
    : statement ';'* EOF
    ;

statement
    : CREATE (OR REPLACE)? MODEL model=qualifiedName
      (FLAVOR flavor=identifier)?
      (PREPROCESSOR preprocess=processorClause)?
      (POSTPROCESSOR postprocess=processorClause)?
      (OPTIONS optionList)?
      (RETURNS datatype=dataType)?
      (USING uri=STRING)	                        # createModel
    | (DESC | DESCRIBE) MODEL model=qualifiedName   # describeModel
    | SHOW MODELS                                   # showModels
    | DROP MODEL model=qualifiedName                # dropModel
    | .*?                                           # passThrough
    ;

qualifiedName: identifier ('.' identifier)*;

identifier
    : IDENTIFIER		# unquotedIdentifier
    | quotedIdentifier	# quotedIdentifierAlternative
    | nonReserved		# unquotedIdentifier
    ;

quotedIdentifier
    : BACKQUOTED_IDENTIFIER
    ;

nonReserved
    : CREATE | DESC | DESCRIBE | MODEL | MODELS | OPTIONS | REPLACE
    ;

ARRAY: A R R A Y;
AS: 'AS';
CREATE: 'CREATE';
DESC : 'DESC';
DESCRIBE : 'DESCRIBE';
DROP: 'DROP';
FALSE: 'FALSE';
FLAVOR: 'FLAVOR';
LIKE: 'LIKE';
MODEL: 'MODEL';
MODELS: 'MODELS';
OPTIONS: 'OPTIONS';
OR: 'OR';
POSTPROCESSOR: 'POSTPROCESSOR';
PREPROCESSOR: 'PREPROCESSOR';
REPLACE: 'REPLACE';
RETURNS: 'RETURNS';
SHOW: 'SHOW';
STRUCT: S T R U C T;
TRUE: 'TRUE';
USING: 'USING';

EQ: '=' | '==';

STRING
    : '\'' ( ~('\''|'\\') | ('\\' .) )* '\''
    | '"' ( ~('"'|'\\') | ('\\' .) )* '"'
    ;

IDENTIFIER
    : (LETTER | DIGIT | '_')+
    ;

BACKQUOTED_IDENTIFIER
    : '`' ( ~'`' | '``' )* '`'
    ;

optionList: '(' option (',' option)* ')';

option
    : key=optionKey EQ value=optionValue
    ;

optionKey
    : qualifiedName
    ;

optionValue
    : INTEGER_VALUE
    | FLOATING_VALUE
    | booleanValue
    | STRING
    ;

processorClause
    : className=STRING
    ;

struct
    : STRUCT '<' field (',' field)* '>'  # structType
    ;

array
    : ARRAY '<' dataType '>'  # arrayType
    ;

dataType
    : struct # nestedStructType
    | array  # nestedArrayType
    | identifier  # plainFieldType
    ;

field
    : name=identifier ':' dataType   # structField
    ;

INTEGER_VALUE
    : MINUS? DIGIT+
    ;

FLOATING_VALUE
    : MINUS? DECIMAL_DIGITS
    ;

booleanValue
    : TRUE | FALSE
    ;

fragment DECIMAL_DIGITS
    : DIGIT+ '.' DIGIT*
    | '.' DIGIT+
    ;

fragment EXPONENT
    : 'E' [+-]? DIGIT+
    ;

fragment DIGIT
    : [0-9]
    ;

fragment LETTER
    : [A-Z]
    ;

SIMPLE_COMMENT
    : '--' ~[\r\n]* '\r'? '\n'? -> channel(HIDDEN)
    ;

BRACKETED_COMMENT
    : '/*' .*? '*/' -> channel(HIDDEN)
    ;

WS  : [ \r\n\t]+ -> channel(HIDDEN)
    ;

UNRECOGNIZED
    : .
    ;

MINUS
    : '-'
    ;

fragment A : [aA];
fragment B : [bB];
fragment C : [cC];
fragment D : [dD];
fragment E : [eE];
fragment F : [fF];
fragment G : [gG];
fragment H : [hH];
fragment I : [iI];
fragment J : [jJ];
fragment K : [kK];
fragment L : [lL];
fragment M : [mM];
fragment N : [nN];
fragment O : [oO];
fragment P : [pP];
fragment Q : [qQ];
fragment R : [rR];
fragment S : [sS];
fragment T : [tT];
fragment U : [uU];
fragment V : [vV];
fragment W : [wW];
fragment X : [xX];
fragment Y : [yY];
fragment Z : [zZ];
