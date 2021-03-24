#  Copyright 2021 Rikai Authors
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


all: antlr

antlr: python/rikai/spark/sql/generated/RikaiModelSchemaParser.py
.PHONY: antlr

python/rikai/spark/sql/generated/RikaiModelSchemaParser.py: src/main/antlr4/org/apache/spark/sql/ml/parser/RikaiModelSchema.g4
	antlr -Dlanguage=Python3 \
		-Xexact-output-dir \
		-no-listener -visitor \
		-o python/rikai/spark/sql/generated \
		src/main/antlr4/org/apache/spark/sql/ml/parser/RikaiModelSchema.g4

lint:
	sbt scalafmtCheckAll
	black -l 79 --check python/rikai python/tests
	pycodestyle --exclude generated python/rikai python/tests
.PHONY: lint

# Fix code style
fix:
	sbt scalafmt
	black -l 79 python/rikai python/tests
.PHONY: fix
