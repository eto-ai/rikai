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

import org.scalatest.funsuite.AnyFunSuite
import ai.eto.rikai.SparkTestSession
import ai.eto.rikai.sql.model.{Catalog, ModelAlreadyExistException}

class CreateModelCommandTest extends AnyFunSuite with SparkTestSession {

  test("create model from uri") {

    spark
      .sql("CREATE MODEL model_created USING 'test://model/created/from/uri'")
      .count()
    assert(Catalog.testing.modelExists("model_created"))

    val model = Catalog.testing.getModel("model_created").get
    assert(model.name == "model_created")
    assert(model.uri == "test://model/created/from/uri")
    assert(model.options.isEmpty)
  }

  test("create model with options") {
    spark
      .sql(
        "CREATE MODEL model_options OPTIONS (foo='bar',num=1.2,flag=True) USING 'test://foo'"
      )
      .count()

    val model = Catalog.testing.getModel("model_options").get
    assert(
      model.options == Seq(
        "foo" -> "bar",
        "num" -> "1.2",
        "flag" -> "true"
      ).toMap
    )
  }

  test("model already exist") {
    spark.sql(
      "CREATE MODEL model_created USING 'test://model/created/from/uri'"
    )
    assertThrows[ModelAlreadyExistException] {
      spark.sql("CREATE MODEL model_created USING 'test://model/other/uri'")
    }
  }
}
