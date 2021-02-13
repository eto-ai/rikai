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

import org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.parser.ParserUtils.withOrigin
import org.apache.spark.sql.catalyst.parser.SqlBaseParser.FunctionCallContext
import org.apache.spark.sql.catalyst.parser.{AstBuilder, ParseException}
import org.apache.spark.sql.ml.catalog.Model
import org.apache.spark.sql.ml.expressions.Predict

import java.util.Locale
import scala.collection.JavaConverters.asScalaBufferConverter

/**
  * Extends Spark's `AstBuilder` to process the `Expression` within
  * SQL Select Clause.
  */
class RikaiSparkAstBuilder extends AstBuilder {

  override def visitFunctionCall(ctx: FunctionCallContext): Expression =
    withOrigin(ctx) {
      ctx.functionName.getText.toLowerCase(Locale.ROOT) match {
        case Predict.name => visitMlPredictFunction(ctx)
        case _            => super.visitFunctionCall(ctx)
      }
    }

  def visitMlPredictFunction(ctx: FunctionCallContext): Expression =
    withOrigin(ctx) {
      val arguments = ctx.argument.asScala.map(expression)
      if (arguments.size < 2) {
        throw new ParseException(
          s"${Predict.name.toUpperCase} requires at least 2 parameters, got ${arguments.size}",
          ctx
        )
      }
      val name = arguments(0) match {
        case arg: UnresolvedAttribute => arg.name
        case _ =>
          throw new ParseException(
            s"Can not recognize model name ${arguments(0)}", ctx
          )
      }
      val model = Model.fromName(name) match {
        case Some(m) => m
        case None => throw new ParseException(s"Could not find model ${name}", ctx)
      }
      model.expression(arguments.drop(1))
    }

}
