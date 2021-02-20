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

package ai.eto.rikai.sql

import org.apache.spark.sql.catalyst.expressions.Expression
import org.scalatest.funsuite.AnyFunSuite

class FakeModel(val name: String, val uri: String) extends Model {

  override def expr(arguments: Seq[Expression]): Expression = ???
}

class CatalogTest extends AnyFunSuite {

  test("Test simple catalog") {
    val catalog = Catalog.testing
    assert(!catalog.modelExists("foo"))
    val created = catalog.createModel(new FakeModel("foo", "bar"))
    assert(created.name == "foo")
    assert(created.uri == "bar")
    assert(catalog.modelExists("foo"))
  }
}
