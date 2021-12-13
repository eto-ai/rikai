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

import java.io.File
import java.net.URI

import ai.eto.rikai.sql.model.{
  Model,
  ModelNotFoundException,
  ModelSpec,
  Registry,
  SparkUDFModel
}
import com.typesafe.scalalogging.LazyLogging

/** [[TestRegistry]] is a Registry for the testing purpose.
  *
  * A valid model URI is: "test://hostname/model_name"
  */
class TestRegistry(conf: Map[String, String])
    extends Registry
    with LazyLogging {

  val schema: String = "test"

  /** Resolve a Model from the specific URI.
    *
    * @param spec Model Spec
    *
    * @return [[Model]] if found, ``None`` otherwise.
    */
  @throws[ModelNotFoundException]
  override def resolve(
      spec: ModelSpec
  ): Model = {
    val parsed = URI.create(spec.uri)
    parsed.getScheme match {
      case this.schema => {
        val model_name: String = spec.name match {
          case Some(name) => name
          case None =>
            new File(
              parsed.getAuthority + "/" + parsed.getPath
            ).getName
        }
        logger.info(s"Creating Spark UDF model: func=${model_name} ${spec}")
        new SparkUDFModel(model_name, spec.uri, model_name, spec.flavor)
      }
      case _ =>
        throw new ModelNotFoundException(s"Fake model ${spec.uri} not found")
    }
  }
}
