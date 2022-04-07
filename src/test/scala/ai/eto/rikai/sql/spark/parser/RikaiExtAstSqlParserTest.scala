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

  test("parse model type") {
    val cmd = parser.parsePlan(
      "CREATE MODEL foo FLAVOR pytorch MODEL_TYPE ssd USING 'abc'"
    )
    assert(cmd.isInstanceOf[CreateModelCommand])
    val create = cmd.asInstanceOf[CreateModelCommand]
    assert(create.modelType.contains("ssd"))
  }

  test("no uri") {
    val cmd = parser
      .parsePlan("""
        |CREATE MODEL resnet18 MODEL_TYPE resnet""".stripMargin)
      .asInstanceOf[CreateModelCommand]
    assert(cmd.name === "resnet18")
    assert(cmd.uri === None)
  }
}
