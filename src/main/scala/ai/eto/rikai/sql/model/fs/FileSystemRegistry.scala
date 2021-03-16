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

import ai.eto.rikai.sql.model.{Model, ModelNotFoundException, Registry}
import ai.eto.rikai.sql.spark.Python
import org.apache.logging.log4j.scala.Logging

/**
  * FileSystem-based [[Registry]].
  */
class FileSystemRegistry(val conf: Map[String, String])
    extends Registry
    with Logging {

  /**
    * Resolve a [[Model]] from the specific URI.
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
  override def resolve(uri: String, name: Option[String]): Model =
    if (uri.endsWith(".yml") || uri.endsWith(".yaml")) {
      logger.info(s"Resolving YAML-based model: ${uri}")
      Python.resolve(uri, name, Map.empty)
    } else if (
      uri.endsWith(".tar") ||
      uri.endsWith(".zip") ||
      uri.endsWith(".tar.gz") ||
      uri.endsWith(".tar.bz")
    ) {
      logger.info(s"Resolving Model tar ball from ${uri}")
      throw new NotImplementedError("Tar-ball Model is not implemented yet.")
    } else {
      throw new ModelUriNotSupportedException(s"URI ${uri} is not supported")
    }
}

class ModelUriNotSupportedException(message: String) extends Exception(message);
