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

import ai.eto.rikai.sql.spark.parser.SchemaParser
import org.apache.spark.sql.rikai.{Box2dType, RikaiTypeRegisters}
import org.apache.spark.sql.types.{
  ArrayType,
  BinaryType,
  BooleanType,
  ByteType,
  DoubleType,
  FloatType,
  IntegerType,
  LongType,
  ShortType,
  StringType,
  StructField,
  StructType
}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

class SchemaParserTest extends AnyFunSuite with BeforeAndAfterAll {

  override def beforeAll(): Unit = {
    super.beforeAll()
    RikaiTypeRegisters.loadUDTs("org.apache.spark.sql")
  }

  test("parse schema") {

    val schema = StructType(
      Seq(
        StructField("id", IntegerType),
        StructField("box", new Box2dType())
      )
    )

    val schemaText = schema.simpleString
    val actual = SchemaParser.parse(schemaText)
    assert(actual == schema)
    assert(actual.simpleString == schema.simpleString)
    assert("struct<id:int,box:box2d>" == schema.simpleString)
  }

  test("parse with nested array") {
    val schema = StructType(
      Seq(
        StructField("id", IntegerType),
        StructField("scores", ArrayType(LongType))
      )
    )

    val schemaText = schema.simpleString
    val actual = SchemaParser.parse(schemaText)
    assert(actual == schema)
    assert(actual.simpleString == schema.simpleString)
    assert("struct<id:int,scores:array<bigint>>" == schema.simpleString)

    // using long also works
    assert(SchemaParser.parse("struct<id:int,scores:array<long>>") == schema)
  }

  test("parse array of floats") {
    val schema = ArrayType(FloatType)
    val schemaText = schema.simpleString
    val actual = SchemaParser.parse(schemaText)

    assert(schema == actual)
  }

  test("parse primitive types") {
    assert(BooleanType.simpleString == "boolean")
    assert(SchemaParser.parse("bool") == BooleanType)
    assert(SchemaParser.parse("boolean") == BooleanType)

    assert("tinyint" == ByteType.simpleString)
    assert(SchemaParser.parse("byte") == ByteType)
    assert(SchemaParser.parse("tinyint") == ByteType)

    assert("smallint" == ShortType.simpleString)
    assert(SchemaParser.parse("smallint") == ShortType)
    assert(SchemaParser.parse("short") == ShortType)

    assert(SchemaParser.parse("int") == IntegerType)

    assert(SchemaParser.parse("double") == DoubleType)

    assert("string" == StringType.simpleString)
    assert(SchemaParser.parse("string") == StringType)

    assert("binary" == BinaryType.simpleString)
    assert(SchemaParser.parse("binary") == BinaryType)
  }
}
