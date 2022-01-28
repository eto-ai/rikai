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

import ai.eto.rikai.sql.spark.execution.CreateModelCommand
import org.apache.spark.sql.catalyst.parser.ParseException
import org.scalatest.funsuite.AnyFunSuite

class RikaiExtAstSqlParserTest extends AnyFunSuite {

  val parser = new RikaiExtSqlParser()

  test("parse create model if not exists") {
    val cmd = parser.parsePlan(
      "CREATE MODEL IF NOT EXISTS foo USING 's3://tmp/test_model'"
    )
    assert(cmd.isInstanceOf[CreateModelCommand])
    val create = cmd.asInstanceOf[CreateModelCommand]
    assert(create.ifNotExists === true)
  }

  test("parse create or replace model if not exists") {
    assertThrows[ParseException] {
      parser.parsePlan("""
          |CREATE OR REPLACE MODEL IF NOT EXISTS model_created
          |USING 'test://model/created/from/uri'
          |""".stripMargin)
    }
  }

  test("parse returns datatype") {
    val cmd = parser.parsePlan(
      "CREATE MODEL foo RETURNS STRUCT<foo:int, bar:ARRAY<STRING>> USING 'abc'"
    )
    assert(cmd.isInstanceOf[CreateModelCommand])
    val create = cmd.asInstanceOf[CreateModelCommand]
    assert(create.name == "foo")
    assert(create.uri.contains("abc"))
    assert(create.returns.contains("STRUCT<foo:int, bar:ARRAY<STRING>>"))
  }

  test("parse model type") {
    val cmd = parser.parsePlan("CREATE MODEL foo FLAVOR pytorch MODEL_TYPE ssd USING 'abc'")
    assert(cmd.isInstanceOf[CreateModelCommand])
    val create = cmd.asInstanceOf[CreateModelCommand]
    assert(create.modelType.contains("ssd"))
  }

  test("parse returns UDTs") {
    val cmd = parser.parsePlan(
      "CREATE MODEL udts RETURNS STRUCT<foo:int, bar:ARRAY<Box2d>> USING 'gs://udt/bucket'"
    )
    assert(cmd.isInstanceOf[CreateModelCommand])
    val create = cmd.asInstanceOf[CreateModelCommand]
    assert(create.name == "udts")
    assert(create.uri.contains("gs://udt/bucket"))
    assert(create.returns.contains("STRUCT<foo:int, bar:ARRAY<Box2d>>"))
  }

  test("parse processors") {
    val cmd = parser
      .parsePlan(
        "CREATE MODEL proc PREPROCESSOR 'rikai.models.yolo.preprocessor' " +
          "USING '/tmp/model'"
      )
      .asInstanceOf[CreateModelCommand]
    val spec = cmd.asSpec
    assert(spec.preprocessor.isDefined)
    assert(spec.preprocessor.contains("rikai.models.yolo.preprocessor"))
    assert(spec.postprocessor.isEmpty)
  }

  test("parse pre&post processors") {
    val cmd = parser
      .parsePlan(
        "CREATE MODEL proc " +
          "PREPROCESSOR 'rikai.models.yolo.preprocessor' " +
          "POSTPROCESSOR \"rikai.models.yolo.postprocessor\"" +
          "USING '/tmp/model'"
      )
      .asInstanceOf[CreateModelCommand]
    val spec = cmd.asSpec
    assert(spec.preprocessor.isDefined)
    assert(spec.preprocessor.contains("rikai.models.yolo.preprocessor"))
    assert(spec.postprocessor.isDefined)
    assert(spec.postprocessor.contains("rikai.models.yolo.postprocessor"))

  }

  test("bad preprocessor") {
    val cmd = parser
      .parsePlan(
        "CREATE MODEL proc PREPROCESSOR ('rikai.models.yolo.preprocessor') " +
          "USING '/tmp/model'"
      )

    assert(cmd == null)
  }
}
