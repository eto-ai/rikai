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

grammar RikaiModelSchema;

schema
    : fieldType
    ;

identifier
    : IDENTIFIER		# unquotedIdentifier
    ;

struct
    : STRUCT '<' field (',' field)* '>'  # structType
    ;

array
    : ARRAY '<' fieldType '>'  # arrayType
    ;

fieldType
    : struct # nestedStructType
    | array  # nestedArrayType
    | identifier  # plainFieldType
    ;

field
    : name=identifier ':' fieldType   # structField
    ;

STRUCT: 'STRUCT';
ARRAY: 'ARRAY';

IDENTIFIER
    : (LETTER | DIGIT | '_')+
    ;

fragment DIGIT
    : [0-9]
    ;

fragment LETTER
    : [A-Za-z]
    ;

WS  : [ \r\n\t]+ -> channel(HIDDEN)
    ;