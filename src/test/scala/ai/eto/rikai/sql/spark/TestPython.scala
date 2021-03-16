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

package ai.eto.rikai.sql.spark

import ai.eto.rikai.sql.model.Model

/**
  * [[Python]] callback service for unit testing
  */
class TestPython extends Python {

  /**
    * Resolve a Model from python.
    *
    * @param uri     URL for a model spec or model file.
    * @param name    Optional model name. Can be empty.
    * @param options options to the model.
    * @return a Model
    */
  override def resolve(
      className: String,
      uri: String,
      name: String,
      options: Map[String, String]
  ): Model = ???
}
