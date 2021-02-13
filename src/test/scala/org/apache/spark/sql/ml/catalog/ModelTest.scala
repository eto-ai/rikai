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

package org.apache.spark.sql.ml.catalog

import org.scalatest.funsuite.AnyFunSuite

class ModelTest extends AnyFunSuite {

  test("create models") {
    val m = new Model("foo", uri = "https://to/foo")
    assert(m.name == "foo")
    assert(m.uri == "https://to/foo")
  }

  test("Parsing URLs") {
    val m = Model.fromName("model.//abc").get
    assert (m.name == "abc")
  }

  test("Parse from local file") {}

  test("Parse from model name") {}
}
