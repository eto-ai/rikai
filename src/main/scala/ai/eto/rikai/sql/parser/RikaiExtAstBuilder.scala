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

import ai.eto.rikai.sql.parser.RikaiSqlBaseParser._
import org.antlr.v4.runtime.tree.{ErrorNode, ParseTree, RuleNode, TerminalNode}
import org.apache.spark.sql.catalyst.parser.ParserUtils.withOrigin
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan

/**
  * ```AstBuilder``` for Rikai Spark SQL extensions.
  */
class RikaiExtAstBuilder extends RikaiSqlBaseVisitor[AnyRef] {

  override def visitSingleStatement(ctx: SingleStatementContext): LogicalPlan =
    withOrigin(ctx) {
      visit(ctx.statement).asInstanceOf[LogicalPlan]
    }

  override def visit(tree: ParseTree): AnyRef = ???

  override def visitCreateModel(ctx: CreateModelContext): LogicalPlan = {
    null
  }

  override def visitPassThrough(ctx: PassThroughContext): LogicalPlan = null

  override def visitQualifiedName(ctx: QualifiedNameContext): AnyRef = ???

  override def visitQuotedIdentifier(ctx: QuotedIdentifierContext): AnyRef = ???

  /**
    * Visit a parse tree produced by {@link RikaiSqlBaseParser# nonReserved}.
    *
    * @param ctx the parse tree
    * @return the visitor result
    */
  override def visitNonReserved(ctx: NonReservedContext): AnyRef = ???

  override def visitChildren(node: RuleNode): AnyRef = ???

  override def visitTerminal(node: TerminalNode): AnyRef = ???

  override def visitErrorNode(node: ErrorNode): AnyRef = ???

  /**
    * Visit a parse tree produced by the {@code unquotedIdentifier}
    * labeled alternative in {@link RikaiSqlBaseParser#   identifier}.
    *
    * @param ctx the parse tree
    * @return the visitor result
    */
  override def visitUnquotedIdentifier(ctx: UnquotedIdentifierContext): AnyRef = ???

  /**
    * Visit a parse tree produced by the {@code quotedIdentifierAlternative}
    * labeled alternative in {@link RikaiSqlBaseParser# identifier}.
    *
    * @param ctx the parse tree
    * @return the visitor result
    */
  override def visitQuotedIdentifierAlternative(ctx: QuotedIdentifierAlternativeContext): AnyRef = ???
}
