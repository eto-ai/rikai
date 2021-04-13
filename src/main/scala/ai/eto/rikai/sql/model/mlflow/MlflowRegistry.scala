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

package ai.eto.rikai.sql.model.mlflow

import ai.eto.rikai.sql.model.{Model, ModelNotFoundException, Registry}
import ai.eto.rikai.sql.spark.Python
import org.apache.logging.log4j.scala.Logging

/** Mlflow-based Model [[Registry]].
  */
class MlflowRegistry(val conf: Map[String, String])
    extends Registry
    with Logging {

  private val pyClass: String =
    "rikai.spark.sql.codegen.mlflow_registry.MlflowRegistry"

  /** Resolve a [[Model]] from the specific URI.
    *
    * @param uri  is the model registry URI.
    * @param name is an optional model name. If provided,
    *             will create the [[Model]] with this name.
    *
    * @throws ModelNotFoundException if the model does not exist on the registry.
    *
    * @return [[Model]] if found.
    */
  @throws[ModelNotFoundException]
  override def resolve(
      uri: String,
      name: Option[String],
      options: Option[Map[String, String]]
  ): Model = {
    logger.info(s"Resolving ML model from ${uri}")
    Python.resolve(pyClass, uri, name, options.getOrElse(Map.empty))
  }
}
