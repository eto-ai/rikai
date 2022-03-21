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

import ai.eto.rikai.sql.model.nouri.NoURIRegistry
import com.typesafe.scalalogging.LazyLogging
import org.apache.http.client.utils.URIUtils
import org.apache.log4j.Logger
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.rikai.model.ModelResolver

import java.net.URI
import scala.util.{Success, Try}

/** Model Registry Integrations.
  */
trait Registry {

  /** Resolve a Model from the specific URI.
    *
    * @param session a live SparkSession.
    * @param spec the spec of a model.
    *
    * @throws ModelNotFoundException if the model does not exist on the registry.
    *
    * @return [[Model]] if found.
    */
  @throws[ModelNotFoundException]
  def resolve(
      session: SparkSession,
      spec: ModelSpec
  ): Model
}

abstract class PyImplRegistry extends Registry with LazyLogging {
  def pyClass: String

  /** Resolve a [[Model]] from the specific URI.
    *
    * @param session a live SparkSession
    * @param spec Model Spec to send to python.
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
    ModelResolver.resolve(session, pyClass, spec)
  }
}

private[rikai] object Registry {

  val REGISTRY_IMPL_PREFIX = "spark.rikai.sql.ml.registry."
  val REGISTRY_IMPL_SUFFIX = ".impl"

  /** To provide convenience when using the CREATE MODEL command we allow the user to specify
    * a URI root at cluster startup. This URI root will automatically be prepended to the specified
    * model spec URI from the USING clause of CREATE MODEL. Note that this is evaluated only
    * when `registerAll` is called so there is no guarantee that changing the value of this
    * config after cluster startup will have any impact.
    */
  val DEFAULT_URI_ROOT_KEY = "spark.rikai.sql.ml.registry.uri.root"

  /** Automatically configure registries for file:/ and mlflow:/ model uri's.
    */
  val DEFAULT_REGISTRIES = Map(
    "spark.rikai.sql.ml.registry.file.impl" -> "ai.eto.rikai.sql.model.fs.FileSystemRegistry",
    "spark.rikai.sql.ml.registry.mlflow.impl" -> "ai.eto.rikai.sql.model.mlflow.MlflowRegistry"
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
        // The user specified default URI root must be a valid URI with a valid scheme
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
  private[model] def getRegistry(uriOption: Option[String]): Registry = {
    uriOption match {
      case Some(uri) =>
        val parsedNormalizedUri = normalize_uri(uri)
        val scheme: String = parsedNormalizedUri.getScheme
        registryMap.get(scheme) match {
          case Some(registry) => registry
          case None =>
            throw new ModelResolveException(
              s"Model registry scheme '${scheme}' is not supported"
            )
        }
      case None =>
        NoURIRegistry
    }
  }

  /** Prepend the default URI root if the input uri string does not have a scheme specified
    * @param uri a user specified uri string
    * @return a URI guaranteed to have a scheme so that we can look up which Registry can handle it
    */
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
      session: SparkSession,
      spec: ModelSpec
  ): Model = getRegistry(spec.uri).resolve(session, spec)

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
