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

/** a FakeModel for testing */
class FakeModel(
    val name: String,
    val uri: String,
    funcName: String,
    val registry: Registry
) extends Model {

  def this(name: String, uri: String, registry: Registry) =
    this(name, uri, name, registry)
}

class FakeRegistry extends Registry {

  /**
    * Resolve a Model from the specific URI.
    *
    * @param uri is the model registry URI
    * @return [[Model]] if found, ``None`` otherwise.
    */
  override def resolve(uri: String): Option[Model] = ???
}
