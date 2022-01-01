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

package ai.eto.rikai.sql.model.fs

import ai.eto.rikai.sql.model.{
  Model,
  ModelNotFoundException,
  ModelSpec,
  Registry
}
import ai.eto.rikai.sql.spark.Python
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.SparkSession

/** FileSystem-based Model [[Registry]].
  */
class FileSystemRegistry(val conf: Map[String, String])
    extends Registry
    with LazyLogging {

  private val pyClass: String = "rikai.spark.sql.codegen.fs.FileSystemRegistry"

  /** Resolve a [[Model]] from the specific URI.
    *
    * @param spec Model Spec of a model
    *
    * @throws ModelNotFoundException if the model does not exist on the registry.
    *
    * @return [[Model]] if found.
    */
  @throws[ModelNotFoundException]
  override def resolve(
      session: SparkSession,
      spec: ModelSpec
  ): Model = {
    logger.info(s"Resolving ML model from ${spec.uri}")
    Python.resolve(session, pyClass, spec)
  }
}
