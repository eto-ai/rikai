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

package ai.eto.rikai.sql.model

import org.scalatest.funsuite.AnyFunSuite

class ModelTest extends AnyFunSuite {

  test("model name matching") {

    assert("abc".matches(Model.namePattern.regex))
    assert("ABC_v123".matches(Model.namePattern.regex))
    assert(!"123_abc".matches(Model.namePattern.regex))
    assert(!"ab@c".matches(Model.namePattern.regex))
  }

  test("model serialize") {
    assert(Model.serializeOptions(Map("foo" -> "bar")) == """{"foo":"bar"}""")
    assert(Model.serializeOptions(Map.empty[String, String]) == "")
  }
}
