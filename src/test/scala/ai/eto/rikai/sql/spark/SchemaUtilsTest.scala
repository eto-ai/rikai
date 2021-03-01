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

import org.apache.spark.sql.rikai.{Box2dType, RikaiTypeRegisters}
import org.apache.spark.sql.types.{
  ArrayType,
  FloatType,
  IntegerType,
  LongType,
  StructField,
  StructType
}
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

class SchemaUtilsTest extends AnyFunSuite with BeforeAndAfterAll {

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
    val actual = SchemaUtils.parse(schemaText)
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
    val actual = SchemaUtils.parse(schemaText)
    assert(actual == schema)
    assert(actual.simpleString == schema.simpleString)
    assert("struct<id:int,scores:array<bigint>>" == schema.simpleString)

    // using long also works
    assert(SchemaUtils.parse("struct<id:int,scores:array<long>>") == schema)
  }

  test("parse array of floats") {
    val schema = ArrayType(FloatType)
    val schemaText = schema.simpleString
    val actual = SchemaUtils.parse(schemaText)

    assert(schema == actual)
  }
}
