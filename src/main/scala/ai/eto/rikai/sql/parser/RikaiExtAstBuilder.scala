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

package ai.eto.rikai.sql.parser

import ai.eto.rikai.sql.execution.{CreateModelCommand,ShowModelsCommand}
import ai.eto.rikai.sql.parser.RikaiSqlBaseParser._
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.catalyst.parser.ParseException
import org.apache.spark.sql.catalyst.parser.ParserUtils.{string, withOrigin}
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan

import scala.collection.JavaConverters.asScalaBufferConverter

<<<<<<< HEAD

=======
>>>>>>> origin/lei/sql_ml
/**
  * ```AstBuilder``` for Rikai Spark SQL extensions.
  */
class RikaiExtAstBuilder extends RikaiSqlBaseBaseVisitor[AnyRef] {

  protected def visitTableIdentfier(
      ctx: QualifiedNameContext
  ): TableIdentifier =
    withOrigin(ctx) {
      ctx.identifier.asScala match {
        case Seq(tbl)     => TableIdentifier(tbl.getText)
        case Seq(db, tbl) => TableIdentifier(tbl.getText, Some(db.getText))
        case _ =>
          throw new ParseException(s"Illegal table name ${ctx.getText}", ctx)
      }
    }

  protected def parseOptionList(ctx: OptionListContext): Map[String, String] =
    withOrigin(ctx) {
      ctx.option().asScala.map(option => (option.key.getText, option.value.getText)).toMap
    }

  override def visitSingleStatement(ctx: SingleStatementContext): LogicalPlan =
    withOrigin(ctx) {
      visit(ctx.statement).asInstanceOf[LogicalPlan]
    }

  override def visitCreateModel(ctx: CreateModelContext): LogicalPlan = {
    CreateModelCommand(
      ctx.model.getText,
      Option(ctx.path).map(string),
      Option(ctx.table).map(visitTableIdentfier),
      replace = false,
      options = parseOptionList(ctx.optionList())
    )
  }

  override def visitShowModels(ctx: ShowModelsContext): LogicalPlan = {
    ShowModelsCommand()
  }

  override def visitPassThrough(ctx: PassThroughContext): AnyRef = null

  override def visitQualifiedName(ctx: QualifiedNameContext): String = {
    println(s"Qualified name: ${ctx}")
    ctx.getText
  }

  override def visitUnquotedIdentifier(
      ctx: UnquotedIdentifierContext
  ): String = {
    println(s"UnquotedIdentifier: ${ctx}")
    ctx.getText
  }

  override def visitQuotedIdentifierAlternative(
      ctx: QuotedIdentifierAlternativeContext
  ): String = ctx.getText

  override def visitQuotedIdentifier(ctx: QuotedIdentifierContext): String =
    ctx.getText

  override def visitNonReserved(ctx: NonReservedContext): AnyRef = ???
}
