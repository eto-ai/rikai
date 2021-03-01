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

package ai.eto.rikai.sql.spark

import ai.eto.rikai.sql.spark.parser.RikaiModelSchemaParser.{
  ArrayTypeContext,
  PlainFieldTypeContext,
  SchemaContext,
  StructFieldContext,
  StructTypeContext
}
import ai.eto.rikai.sql.spark.parser.{
  RikaiModelSchemaBaseVisitor,
  RikaiModelSchemaLexer,
  RikaiModelSchemaParser
}
import org.antlr.v4.runtime.atn.PredictionMode
import org.antlr.v4.runtime.{CharStreams, CommonTokenStream}
import org.apache.spark.sql.rikai.RikaiTypeRegisters
import org.apache.spark.sql.types._

import scala.collection.JavaConverters.iterableAsScalaIterableConverter

class RikaiModelSchemaBuilder extends RikaiModelSchemaBaseVisitor[AnyRef] {
  override def visitSchema(
      ctx: SchemaContext
  ): DataType = {
    visit(ctx.struct).asInstanceOf[DataType]
  }

  override def visitStructType(ctx: StructTypeContext): StructType = {
    new StructType(
      ctx.field.asScala.map(visit).map(v => v.asInstanceOf[StructField]).toArray
    )
  }

  override def visitStructField(ctx: StructFieldContext): StructField = {
    StructField(
      ctx.name.getText,
      visit(ctx.fieldType).asInstanceOf[DataType]
    )
  }

  override def visitArrayType(ctx: ArrayTypeContext): ArrayType = {
    ArrayType(visit(ctx.fieldType()).asInstanceOf[DataType])
  }

  override def visitPlainFieldType(ctx: PlainFieldTypeContext): DataType = {
    val typeName = ctx.identifier().getText
    typeName.toLowerCase match {
      case "int"             => IntegerType
      case "long" | "bigint" => LongType
      case "float"           => FloatType
      case "double"          => DoubleType
      case _ => {
        RikaiTypeRegisters.get(typeName) match {
          case Some(dt) => dt.getDeclaredConstructor().newInstance()
          case None =>
            throw new UnrecognizedSchema(s"Dose not recognize ${typeName}")
        }
      }
    }
  }
}

object SchemaUtils {

  private val builder = new RikaiModelSchemaBuilder()

  /** Parse a schema simpleString with UDTs */
  def parse(schema: String): DataType = {
    parseSchema(schema) { parser =>
      {
        builder.visitSchema(parser.schema()) match {
          case dataType: DataType => dataType
          case _ =>
            throw new RuntimeException(s"does not recognize schema: ${schema}")
        }
      }
    }
  }

  private def parseSchema[T](
      schema: String
  )(toResult: RikaiModelSchemaParser => T): T = {
    val lexer = new RikaiModelSchemaLexer(CharStreams.fromString(schema))
    val tokenStream = new CommonTokenStream(lexer)
    val parser = new RikaiModelSchemaParser(tokenStream)
    parser.getInterpreter.setPredictionMode(PredictionMode.SLL)
    toResult(parser)
  }
}

class UnrecognizedSchema(message: String) extends Exception(message)
