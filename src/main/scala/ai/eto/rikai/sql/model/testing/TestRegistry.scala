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

package ai.eto.rikai.sql.model.testing

import ai.eto.rikai.sql.model.{Model, ModelNotFoundException, Registry}

import java.io.File
import java.net.URI

/**
  * [[TestRegistry]] is a Registry for the testing purpose.
  *
  * A valid model URI is: "test://hostname/model_name"
  */
class TestRegistry(conf: Map[String, String]) extends Registry {

  val schema: String = "test"

  /**
    * Resolve a Model from the specific URI.
    *
    * @param uri is the model registry URI.
    *
    * @return [[Model]] if found, ``None`` otherwise.
    */
  @throws[ModelNotFoundException]
  override def resolve(uri: String, name: Option[String] = None): Model = {
    val parsed = URI.create(uri)
    parsed.getScheme match {
      case this.schema => {
        val model_name = name match {
          case Some(name) => name
          case None =>
            new File(
              parsed.getAuthority + "/" + parsed.getPath
            ).getName
        }
        new TestModel(model_name, uri, this)
      }
      case _ => throw new ModelNotFoundException(s"Fake model ${uri} not found")
    }
  }
}
