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

import ai.eto.rikai.sql.expressions.Predict
import ai.eto.rikai.sql.model.{Catalog, Registry}
import ai.eto.rikai.sql.spark.SparkRunnable
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.analysis.UnresolvedAttribute
import org.apache.spark.sql.catalyst.expressions.{Expression, Literal}
import org.apache.spark.sql.catalyst.parser.ParserUtils.withOrigin
import org.apache.spark.sql.catalyst.parser.SqlBaseParser.FunctionCallContext
import org.apache.spark.sql.catalyst.parser.{AstBuilder, ParseException}

import java.util.Locale
import scala.collection.JavaConverters.asScalaBufferConverter

/**
  * Extends Spark's `AstBuilder` to process the `Expression` within
  * Spark SQL Select/Where/OrderBy clauses.
  */
private[parser] class RikaiSparkSQLAstBuilder(session: SparkSession)
    extends AstBuilder {

  val catalog: Catalog =
    Catalog.getOrCreate(
      session.conf.get(
        Catalog.SQL_ML_CATALOG_IMPL_KEY,
        Catalog.SQL_ML_CATALOG_IMPL_DEFAULT
      )
    )

  val registry: Registry = Registry.get(
    session.conf.get(Registry.MODEL_REGISTRY_IMPL_KEY)
  )

  override def visitFunctionCall(ctx: FunctionCallContext): Expression =
    withOrigin(ctx) {
      ctx.functionName.getText.toLowerCase(Locale.ROOT) match {
        case Predict.name => visitMlPredictFunction(ctx)
        case _            => super.visitFunctionCall(ctx)
      }
    }

  /**
    * Process `ML_PREDICT` Expression.
    */
  def visitMlPredictFunction(ctx: FunctionCallContext): Expression =
    withOrigin(ctx) {
      val arguments = ctx.argument.asScala.map(expression)
      if (arguments.size < 2) {
        throw new ParseException(
          s"${Predict.name.toUpperCase} requires at least 2 parameters, got ${arguments.size}",
          ctx
        )
      }

      val model = arguments.head match {
        case arg: UnresolvedAttribute => catalog.getModel(arg.name)
        case arg: Literal             => registry.resolve(arg.toString)
        case _ =>
          throw new ParseException(
            s"Can not recognize model name ${arguments.head}, class=${arguments.head.getClass}",
            ctx
          )
      }

      model match {
        case Some(runnable) =>
          runnable match {
            case r: SparkRunnable => r.asSpark(arguments.drop(1))
            case _ =>
              throw new ParseException(
                s"Model ${model} is not runnable in Spark",
                ctx
              )
          }
        case None =>
          throw new ParseException(
            s"Model ${arguments.head} does not exist",
            ctx
          )
      }
    }

}
