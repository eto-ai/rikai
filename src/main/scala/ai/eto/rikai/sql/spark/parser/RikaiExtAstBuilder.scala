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

import ai.eto.rikai.sql.model.Model
import ai.eto.rikai.sql.spark.execution.{
  CreateModelCommand,
  DescribeModelCommand,
  DropModelCommand,
  ShowModelsCommand
}
import ai.eto.rikai.sql.spark.parser.RikaiExtSqlBaseParser._
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.catalyst.parser.ParseException
import org.apache.spark.sql.catalyst.parser.ParserUtils.{string, withOrigin}
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan

import scala.collection.JavaConverters.asScalaBufferConverter

/** ```AstBuilder``` for Rikai Spark SQL extensions.
  */
private[parser] class RikaiExtAstBuilder
    extends RikaiExtSqlBaseBaseVisitor[AnyRef] {

  override def visitSingleStatement(ctx: SingleStatementContext): LogicalPlan =
    withOrigin(ctx) {
      visit(ctx.statement).asInstanceOf[LogicalPlan]
    }

  override def visitPassThrough(ctx: PassThroughContext): AnyRef = null

  override def visitCreateModel(ctx: CreateModelContext): LogicalPlan = {
    Model.verifyName(ctx.model.getText)

    val flavor: Option[String] = ctx.flavor match {
      case null => None
      case _ =>
        visit(ctx.flavor) match {
          case s: String => Some(s)
          case _         => None
        }
    }
    val returns: Option[String] = ctx.RETURNS() match {
      case null => None
      case _ =>
        visit(ctx.datatype) match {
          case r: String => Some(r)
          case _ =>
            throw new ParseException(
              s"Expect string type for RETURNS, got ${visit(ctx.datatype)}",
              ctx
            )
        }
    }
    val preprocessor: Option[String] =
      Option(ctx.preprocess).map(visitProcessorClause) match {
        case Some(p: String) => Some(p)
        case _               => None
      }
    val postprocessor: Option[String] =
      Option(ctx.postprocess).map(visitProcessorClause) match {
        case Some(p: String) => Some(p)
        case _               => None
      }

    CreateModelCommand(
      ctx.model.getText,
      flavor = flavor,
      returns = returns,
      uri = Option(ctx.uri).map(string),
      preprocessor = preprocessor,
      postprocessor = postprocessor,
      table = None,
      replace = false,
      options = visitOptionList(ctx.optionList())
    )
  }

  override def visitOptionList(ctx: OptionListContext): Map[String, String] =
    ctx match {
      case null => Map.empty
      case _    => ctx.option().asScala.map(visitOption).toMap
    }

  override def visitOption(ctx: OptionContext): (String, String) = {
    // TODO: find a more scala way?
    val value: String = if (ctx.value.booleanValue() != null) {
      if (ctx.value.booleanValue().TRUE() != null)
        "true"
      else {
        "false"
      }
    } else if (ctx.value.FLOATING_VALUE() != null) {
      ctx.value.FLOATING_VALUE.getSymbol.getText
    } else if (ctx.value.INTEGER_VALUE() != null) {
      ctx.value.INTEGER_VALUE.getSymbol.getText
    } else {
      string(ctx.value.STRING)
    }

    visitQualifiedName(ctx.key.qualifiedName()) -> value
  }

  override def visitQualifiedName(ctx: QualifiedNameContext): String =
    ctx.getText

  override def visitDescribeModel(ctx: DescribeModelContext): LogicalPlan = {
    Model.verifyName(ctx.model.getText)
    DescribeModelCommand(ctx.model.getText)
  }

  override def visitDropModel(ctx: DropModelContext): LogicalPlan = {
    Model.verifyName(ctx.model.getText)
    DropModelCommand(
      ctx.model.getText
    )
  }

  override def visitShowModels(ctx: ShowModelsContext): LogicalPlan = {
    ShowModelsCommand()
  }

  override def visitUnquotedIdentifier(
      ctx: UnquotedIdentifierContext
  ): String = ctx.getText

  override def visitQuotedIdentifierAlternative(
      ctx: QuotedIdentifierAlternativeContext
  ): String = ctx.getText

  override def visitQuotedIdentifier(ctx: QuotedIdentifierContext): String =
    ctx.getText

  override def visitNonReserved(ctx: NonReservedContext): AnyRef = ???

  override def visitArrayType(ctx: ArrayTypeContext): String =
    s"ARRAY<${visit(ctx.dataType)}>"

  override def visitStructType(ctx: StructTypeContext): String =
    s"STRUCT<${ctx.field().asScala.map(visit).mkString(", ")}>"

  override def visitStructField(ctx: StructFieldContext): String =
    s"${ctx.identifier().getText}:${visit(ctx.dataType())}"

  override def visitPlainFieldType(ctx: PlainFieldTypeContext): String =
    ctx.getText

  override def visitProcessorClause(ctx: ProcessorClauseContext): String =
    ctx.className.getText.replaceAll("^[\"']+|[\"']+$", "")

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

}
