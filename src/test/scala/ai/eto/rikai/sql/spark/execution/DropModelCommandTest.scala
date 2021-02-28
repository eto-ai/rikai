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
import ai.eto.rikai.sql.model.testing.TestModel
import ai.eto.rikai.sql.model.{Catalog, ModelNotFoundException, SimpleCatalog}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.parser.ParseException
import org.scalatest.funsuite.AnyFunSuite

class DropModelCommandTest extends AnyFunSuite with SparkTestSession {

  def getCatalog(session: SparkSession): SimpleCatalog =
    Catalog
      .getOrCreate(session.conf.get(Catalog.SQL_ML_CATALOG_IMPL_KEY))
      .asInstanceOf[SimpleCatalog]

  test("test drop one model") {
    val catalog = getCatalog(spark)
    assert(!catalog.modelExists("dropped_model"))

    catalog.createModel(new TestModel("dropped_model", "uri", null))
    assert(catalog.modelExists("dropped_model"))

    spark.sql("DROP MODEL dropped_model").show()
    assert(!catalog.modelExists("dropped_model"))
  }

  test("drop not exist table") {
    assertThrows[ModelNotFoundException] {
      spark.sql("DROP MODEL not_exist").show()
    }
  }

  test("drop table bad name") {
    assertThrows[ParseException] {
      spark.sql("DROP MODEL").count()
    }
  }
}
