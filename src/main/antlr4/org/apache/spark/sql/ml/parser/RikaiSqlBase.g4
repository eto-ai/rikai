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
grammar RikaiSqlBase;

singleStatement
    : statement ';'* EOF
    ;

statement:
	CREATE (OR REPLACE)? MODEL model=qualifiedName
	(OPTIONS '(' ')')?
	(AS table=qualifiedName | USING path=STRING)	# createModel
	| .*?									        # passThrough
	;

qualifiedName: identifier ('.' identifier)*;

identifier:
	IDENTIFIER			# unquotedIdentifier
	| quotedIdentifier	# quotedIdentifierAlternative
	| nonReserved		# unquotedIdentifier
	;

quotedIdentifier:
    BACKQUOTED_IDENTIFIER
    ;

nonReserved:
    ;

AS: 'AS';
CREATE: 'CREATE';
MODEL: 'MODEL';
OPTIONS: 'OPTIONS';
OR: 'OR';
REPLACE: 'REPLACE';
USING: 'USING';

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