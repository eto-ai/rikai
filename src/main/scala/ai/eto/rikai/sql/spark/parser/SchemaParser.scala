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

import ai.eto.rikai.sql.spark.parser.RikaiModelSchemaParser.{
  ArrayTypeContext,
  PlainFieldTypeContext,
  StructFieldContext,
  StructTypeContext
}
import org.antlr.v4.runtime.atn.PredictionMode
import org.antlr.v4.runtime.{CharStreams, CommonTokenStream}
import org.apache.spark.sql.rikai.UDTCollection
import org.apache.spark.sql.types._

import scala.collection.JavaConverters.iterableAsScalaIterableConverter

/**
  * Build Spark's DataType from the model schema AST.
  *
  * It accepts any forms of value from `DataType.simpleString()`:
  *
  * {{{
  *   * struct<score:int, box:box2d, label:string>
  *   * struct<id:int, detection:array<struct<box:box2d, class:int>>>
  *   * array<float>
  *   * Primitive type: float
  * }}}
  */
private class SchemaBuilder extends RikaiModelSchemaBaseVisitor[AnyRef] {

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
      case "boolean" | "bool"   => BooleanType
      case "byte" | "tinyint"   => ByteType
      case "short" | "smallint" => ShortType
      case "int"                => IntegerType
      case "long" | "bigint"    => LongType
      case "float"              => FloatType
      case "double"             => DoubleType
      case "string"             => StringType
      case "binary"             => BinaryType
      case _ => {
        UDTCollection.get(typeName) match {
          case Some(dt) => dt.getDeclaredConstructor().newInstance()
          case None =>
            throw new SchemaParseException(s"Dose not recognize ${typeName}")
        }
      }
    }
  }
}

private[sql] object SchemaParser {

  private val builder = new SchemaBuilder()

  /** Parse a schema simpleString with UDTs */
  def parse(schema: String): DataType = {
    parseSchema(schema) { parser =>
      {
        builder.visit(parser.schema()) match {
          case dataType: DataType => dataType
          case _ =>
            throw new SchemaParseException(
              s"does not recognize schema: ${schema}"
            )
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

class SchemaParseException(message: String) extends Exception(message)
