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

package ai.eto.rikai.sql.spark.execution

import ai.eto.rikai.SparkTestSession
import ai.eto.rikai.sql.model.{Catalog, ModelNotFoundException}
import org.scalatest.funsuite.AnyFunSuite

class DescribeModelCommandTest extends AnyFunSuite with SparkTestSession {

  import spark.implicits._

  test("describe model") {
    Catalog.testing.createModel(
      new TestModel("model_foo", "uri://model/foo", null)
    )

    val expected = Seq(("model_foo", "uri://model/foo", "")).toDF(
      "model",
      "uri",
      "options"
    )
    assertEqual(spark.sql("DESCRIBE MODEL model_foo"), expected)
    assertEqual(spark.sql("DESC MODEL model_foo"), expected)
  }

  test("describe non-exist model") {
    assertThrows[ModelNotFoundException] {
      spark.sql("DESCRIBE MODEL not_exist")
    }
  }
}
