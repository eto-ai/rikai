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

import ai.eto.rikai.sql.model.Registry
import com.thoughtworks.enableIf
import com.thoughtworks.enableIf.classpathMatches
import org.antlr.v4.runtime._
import org.antlr.v4.runtime.atn.PredictionMode
import org.antlr.v4.runtime.misc.{Interval, ParseCancellationException}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.parser.{
  ParseErrorListener,
  ParseException,
  ParserInterface,
  PostProcessor
}
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.catalyst.trees.Origin
import org.apache.spark.sql.catalyst.{FunctionIdentifier, TableIdentifier}
import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.sql.{AnalysisException, SparkSession}

import java.util.concurrent.atomic.AtomicBoolean

/** SQL Parser for Rikai Extensions.
  *
  * Rikai extends Spark SQL to support ML-related SQL DDL.
  *
  * @param session SparkSession
  * @param delegate the fallback parser if this can not recognize the statement.
  */
private[spark] class RikaiExtSqlParser(
    val session: SparkSession,
    val delegate: ParserInterface,
    val testing: Boolean = false
) extends ParserInterface
    with Logging {

  /** Used for test only */
  def this() = {
    this(null, null, true)
  }

  private val builder = new RikaiExtAstBuilder()

  override def parsePlan(sqlText: String): LogicalPlan = {
    import RikaiExtSqlParser._

    if (!testing && !registryInitialized.get()) {
      initRegistry(session)
    }
    parse(sqlText) { parser =>
      {
        builder.visit(parser.singleStatement) match {
          case plan: LogicalPlan => plan
          case _ =>
            if (delegate != null) { delegate.parsePlan(sqlText) }
            else {
              if (!testing)
                throw new RuntimeException("We should only reach here in test")
              return null
            }
        }
      }
    }
  }

  protected def parse[T](
      command: String
  )(toResult: RikaiExtSqlBaseParser => T): T = {
    val lexer = new RikaiExtSqlBaseLexer(
      new UpperCaseCharStream(CharStreams.fromString(command))
    )
    lexer.removeErrorListeners()
    lexer.addErrorListener(ParseErrorListener)

    val tokenStream = new CommonTokenStream(lexer)
    val parser = new RikaiExtSqlBaseParser(tokenStream)
    parser.addParseListener(PostProcessor)
    parser.removeErrorListeners()
    parser.addErrorListener(ParseErrorListener)

    try {
      try {
        // first, try parsing with potentially faster SLL mode
        parser.getInterpreter.setPredictionMode(PredictionMode.SLL)
        toResult(parser)
      } catch {
        case e: ParseCancellationException =>
          // if we fail, parse with LL mode
          tokenStream.seek(0) // rewind input stream
          parser.reset()

          // Try Again.
          parser.getInterpreter.setPredictionMode(PredictionMode.LL)
          toResult(parser)
      }
    } catch {
      case e: ParseException if e.command.isDefined =>
        throw e
      case e: ParseException =>
        throw e.withCommand(command)
      case e: AnalysisException =>
        val position = Origin(e.line, e.startPosition)
        throw new ParseException(Option(command), e.message, position, position)
    }
  }

  override def parseExpression(sqlText: String): Expression =
    delegate.parseExpression(sqlText)

  override def parseTableIdentifier(sqlText: String): TableIdentifier =
    delegate.parseTableIdentifier(sqlText)

  override def parseFunctionIdentifier(sqlText: String): FunctionIdentifier =
    delegate.parseFunctionIdentifier(sqlText)

  override def parseMultipartIdentifier(sqlText: String): Seq[String] =
    delegate.parseMultipartIdentifier(sqlText)

  override def parseTableSchema(sqlText: String): StructType =
    delegate.parseTableSchema(sqlText)

  override def parseDataType(sqlText: String): DataType =
    delegate.parseDataType(sqlText)

  @enableIf(classpathMatches(".*spark-catalyst_2\\.\\d+-3\\.[^012]\\..*".r))
  override def parseQuery(sqlText: String): LogicalPlan =
    delegate.parseQuery(sqlText)
}

private[spark] object RikaiExtSqlParser {
  private val registryInitialized = new AtomicBoolean(false)

  private def runOnce(func: => Unit): Unit = {
    if (registryInitialized.compareAndSet(false, true)) {
      func
    }
  }

  def initRegistry(session: SparkSession): Unit = {
    runOnce {
      Registry.registerAll(session.conf.getAll)
    }
  }
}

// scalastyle:off line.size.limit
/** Fork from `org.apache.spark.sql.catalyst.parser.UpperCaseCharStream`.
  *
  * @see https://github.com/apache/spark/blob/v2.4.4/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/parser/ParseDriver.scala#L157
  */
// scalastyle:on
class UpperCaseCharStream(wrapped: CodePointCharStream) extends CharStream {
  override def consume(): Unit = wrapped.consume
  override def getSourceName(): String = wrapped.getSourceName
  override def index(): Int = wrapped.index
  override def mark(): Int = wrapped.mark
  override def release(marker: Int): Unit = wrapped.release(marker)
  override def seek(where: Int): Unit = wrapped.seek(where)

  override def getText(interval: Interval): String = {
    // ANTLR 4.7's CodePointCharStream implementations have bugs when
    // getText() is called with an empty stream, or intervals where
    // the start > end. See
    // https://github.com/antlr/antlr4/commit/ac9f7530 for one fix
    // that is not yet in a released ANTLR artifact.
    if (size() > 0 && (interval.b - interval.a >= 0)) {
      wrapped.getText(interval)
    } else {
      ""
    }
  }

  override def size(): Int = wrapped.size

  override def LA(i: Int): Int = {
    val la = wrapped.LA(i)
    if (la == 0 || la == IntStream.EOF) la
    else Character.toUpperCase(la)
  }
}
