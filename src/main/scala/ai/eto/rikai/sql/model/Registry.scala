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

import org.apache.log4j.Logger

import java.net.URI

/**
  * Model Registry Integrations.
  */
trait Registry {

  val conf: Map[String, String]

  /**
    * Resolve a Model from the specific URI.
    *
    * @param uri is the model registry URI.
    * @param name is an optional model name. If provided,
    *             will create the [[Model]] with this name.
    *
    * @throws ModelNotFoundException if the model does not exist on the registry.
    *
    * @return [[Model]] if found.
    */
  @throws[ModelNotFoundException]
  def resolve(uri: String, name: Option[String] = None): Model
}

object Registry {

  private val logger = Logger.getLogger(Registry.getClass)

  val REGISTRY_IMPL_PREFIX = "rikai.sql.ml.registry."
  val REGISTRY_IMPL_SUFFIX = ".impl"

  /** Mapping from Model URI schema to the registry. */
  private var registryMap: Map[String, Registry] = Map.empty

  @throws[ModelResolveException]
  private def verifySchema(schema: String): Unit = {
    val schemaRegex = """[a-zA-Z][\w\+]{0,255}"""
    if (!schema.matches(schemaRegex)) {
      throw new ModelResolveException(
        s"Schema '${schema}' does not match '${schemaRegex}'"
      )
    }
  }

  /**
    * Register all registry implementations.
    *
    * @param conf a mapping of (key, value) pairs
    */
  def registerAll(conf: Map[String, String]): Unit = {
    for ((key, value) <- conf) {
      if (
        key.startsWith(REGISTRY_IMPL_PREFIX) &&
        key.endsWith(REGISTRY_IMPL_SUFFIX)
      ) {
        val schema =
          key.substring(
            REGISTRY_IMPL_PREFIX.length,
            key.length - REGISTRY_IMPL_SUFFIX.length
          )
        verifySchema(schema)
        if (registryMap.contains(schema)) {
          throw new ModelRegistryAlreadyExistException(
            s"ModelRegistry(${schema}) exists"
          )
        }
        registryMap += (schema ->
          Class
            .forName(value)
            .getDeclaredConstructor(classOf[Map[String, String]])
            .newInstance(conf)
            .asInstanceOf[Registry])
        logger.debug(s"Model Registry ${schema} registered to: ${value}")
      }

    }
  }

  /**
    * Resolve a [[Model]] from a model registry URI.
    *
    * Internally it uses model registry URI to find the appropriated [[Registry]] to run
    * [[Registry.resolve]].
    *
    * @param uri the model registry URI
    * @param name optionally, model name
    *
    * @throws ModelNotFoundException if the model not found on the registry.
    * @throws ModelResolveException can not resolve the model due to system issues.
    *
    * @return the resolved [[Model]]
    */
  @throws[ModelResolveException]
  @throws[ModelNotFoundException]
  def resolve(uri: String, name: Option[String] = None): Model = {
    val parsedUri = new URI(uri)
    val schema = parsedUri.getScheme
    registryMap.get(schema) match {
      case Some(registry) => registry.resolve(uri, name = name)
      case None =>
        throw new ModelResolveException(
          s"Model registry schema '${schema}' is not supported"
        )
    }
  }

  /** Used in testing to reset the registry */
  private[sql] def reset: Unit = {
    registryMap = Map.empty
  }
}

class ModelResolveException(message: String) extends Exception(message)

class ModelNotFoundException(message: String) extends Exception(message)

class ModelRegistryAlreadyExistException(message: String)
    extends Exception(message)
