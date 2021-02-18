/*
 * Copyright 2021 Rikai authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.ml.parser

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.parser.{
  AbstractSqlParser,
  AstBuilder,
  ParserInterface
}
import org.apache.spark.sql.internal.SQLConf

/**
  * Rikai-extended Spark SQL Parser
  *
  * @param session Live Spark Session
  * @param delegate the delegated Spark SQL parser.
  *
  */
class RikaiSparkSQLParser(session: SparkSession, delegate: ParserInterface)
    extends AbstractSqlParser(SQLConf.getFallbackConf) {

  override protected def astBuilder: AstBuilder = new RikaiSparkAstBuilder()
}
