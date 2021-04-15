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

import org.apache.http.client.utils.URIUtils
import org.apache.log4j.Logger
import java.net.URI

import scala.util.{Success, Try}

/** Model Registry Integrations.
  */
trait Registry {

  /** Resolve a Model from the specific URI.
    *
    * @param spec the spec of a model.
    *
    * @throws ModelNotFoundException if the model does not exist on the registry.
    *
    * @return [[Model]] if found.
    */
  @throws[ModelNotFoundException]
  def resolve(
      spec: ModelSpec
  ): Model
}

private[rikai] object Registry {

  val REGISTRY_IMPL_PREFIX = "rikai.sql.ml.registry."
  val REGISTRY_IMPL_SUFFIX = ".impl"
  val DEFAULT_URI_ROOT_KEY = "rikai.sql.ml.registry.uri.root"
  val DEFAULT_REGISTRIES = Map(
    "rikai.sql.ml.registry.file.impl" -> "ai.eto.rikai.sql.model.fs.FileSystemRegistry",
    "rikai.sql.ml.registry.mlflow.impl" -> "ai.eto.rikai.sql.model.mlflow.MlflowRegistry"
  )
  private val logger = Logger.getLogger(Registry.getClass)

  /** Mapping from Model URI scheme to the registry. */
  private var registryMap: Map[String, Registry] = Map.empty

  /** by default we assume the Uri is for the local file system */
  private var defaultUriRoot: URI = new URI("file:/")

  /** Register all registry implementations.
    *
    * @param conf a mapping of (key, value) pairs
    */
  def registerAll(conf: Map[String, String]): Unit = {
    // defaults
    for ((key, value) <- DEFAULT_REGISTRIES ++ conf) {
      if (key == DEFAULT_URI_ROOT_KEY) {
        Try(new URI(value)) match {
          case Success(uri)
              if (uri.getScheme != null && uri.getScheme.nonEmpty) => {
            defaultUriRoot = uri
          }
          case _ =>
            throw new IllegalArgumentException(
              s"Default URI root $value is not well-formed or does not specify a scheme"
            )
        }
      } else if (
        key.startsWith(REGISTRY_IMPL_PREFIX) &&
        key.endsWith(REGISTRY_IMPL_SUFFIX)
      ) {
        val scheme: String = key.substring(
          REGISTRY_IMPL_PREFIX.length,
          key.length - REGISTRY_IMPL_SUFFIX.length
        )
        verifyScheme(scheme)
        if (registryMap.contains(scheme))
          throw new ModelRegistryAlreadyExistException(
            if (scheme != null) { s"ModelRegistry(${scheme}) exists" }
            else { s"Default ModelRegistry exists" }
          )
        registryMap += (scheme ->
          Class
            .forName(value)
            .getDeclaredConstructor(classOf[Map[String, String]])
            .newInstance(conf)
            .asInstanceOf[Registry])
        logger.debug(s"Model Registry ${scheme} registered to: ${value}")
      }
    }
  }

  @throws[ModelResolveException]
  private def verifyScheme(scheme: String): Unit = {
    val schemeRegex = """[a-zA-Z][\w\+]{0,255}"""
    if (scheme != null && !scheme.matches(schemeRegex)) {
      throw new ModelResolveException(
        s"Scheme '${scheme}' does not match '${schemeRegex}'"
      )
    }
  }

  @throws[ModelResolveException]
  private[model] def getRegistry(uri: String): Registry = {
    val parsedNormalizedUri = normalize_uri(uri)
    val scheme: String = parsedNormalizedUri.getScheme
    registryMap.get(scheme) match {
      case Some(registry) => registry
      case None =>
        throw new ModelResolveException(
          s"Model registry scheme '${scheme}' is not supported"
        )
    }
  }

  private[sql] def normalize_uri(uri: String): URI = {
    val parsedUri = new URI(uri)
    parsedUri.getScheme match {
      case s: String if (!s.isEmpty) => parsedUri
      case _                         => URIUtils.resolve(defaultUriRoot, uri)
    }
  }

  /** Resolve a [[Model]] from a model registry URI.
    *
    * Internally it uses model registry URI to find the appropriated [[Registry]] to run
    * [[Registry.resolve]].
    *
    * @param spec Model Spec.
    *
    * @throws ModelNotFoundException if the model not found on the registry.
    * @throws ModelResolveException can not resolve the model due to system issues.
    *
    * @return the resolved [[Model]]
    */
  @throws[ModelResolveException]
  @throws[ModelNotFoundException]
  def resolve(
      spec: ModelSpec
  ): Model = getRegistry(spec.uri).resolve(spec)

  /** Used in testing to reset the registry */
  private[sql] def reset: Unit = {
    registryMap = Map.empty
  }
}

class ModelResolveException(message: String) extends Exception(message)

class ModelNotFoundException(message: String) extends Exception(message)

class ModelRegistryAlreadyExistException(message: String)
    extends Exception(message)

class ModelAlreadyExistException(message: String) extends Exception(message)
