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

package ai.eto.rikai.sql.spark.parser

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.parser.{
  AbstractSqlParser,
  AstBuilder,
  ParserInterface
}

/**
  * Spark SQL-ML Parser.
  *
  * This parser injects "ML_PREDICT" as a FUNCTION expression, and compiles a LogicalPlan that
  * fits for later implementation-specific model inference physical plan.
  *
  * Instead of building a separate SQL parser like [[RikaiExtSqlParser]], we extend the actual
  * Spark's catalylst SQLParser and AstBuilder, so that ``ML_PREDICT`` can work as part of the
  * pure Spark SQL statements.
  *
  * For example:
  * {{{
  *   SELECT
  *    id, ground_label, ML_PREDICT(new_model, image) as detections
  *   FROM images
  *   WHERE
  *    ground_label != ML_PREDICT(old_model, image).label
  * }}}
  *
  * @param context Contains Rikai runtime information
  * @param delegate the delegated Spark SQL parser.
  */
private[sql] class RikaiSparkSQLParser(
    session: SparkSession,
    delegate: ParserInterface
) extends AbstractSqlParser {

  override protected def astBuilder: AstBuilder =
    new RikaiSparkSQLAstBuilder(session)
}
