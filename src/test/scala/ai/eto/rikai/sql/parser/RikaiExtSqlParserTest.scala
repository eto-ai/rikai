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

import ai.eto.rikai.sql.execution.{CreateModelCommand, ShowModelsCommand}
import org.apache.spark.sql.ml.parser.RikaiSparkSQLParser
import org.scalatest.funsuite.AnyFunSuite

class RikaiExtSqlParserTest extends AnyFunSuite {

  val parser = new RikaiExtSqlParser(new RikaiSparkSQLParser(null, null))

  test("Test parse CREATE MODEL using external URL") {
    val plan = parser.parsePlan("CREATE MODEL foo USING 'model://foo/bar'")
    plan match {
      case cmd: CreateModelCommand => {
        assert(cmd.name == "foo")
        assert(cmd.path.get == "model://foo/bar")
      }
      case _ => {
        assert(
          false,
          s"Expect a CreateModelCommand, but got ${plan.getClass}"
        )
      }
    }
  }

  test("Test case insensitive") {
    assert(
      parser
        .parsePlan("CreaTe ModEL foo UsinG 'a/b/c.zip'")
        .isInstanceOf[CreateModelCommand]
    )
  }

  test("Test parse SHOW MODELS") {
    assert(
      parser
       .parsePlan("SHOW MODELS")
       .isInstanceOf[ShowModelsCommand]
    )
  }
}
