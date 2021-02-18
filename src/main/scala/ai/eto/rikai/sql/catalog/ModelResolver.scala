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

package ai.eto.rikai.sql.catalog

import org.apache.spark.sql.SparkSession

/**
  * Model Resolver trait.
  *
  * Resolve a Model from a thrid party model registry.
  */
trait ModelResolver {

  /**
    * Resolve a Model from a URI.
    *
    * @param uri Model URI
    * @return a resolved model.
    */
  def resolve(uri: String): Option[Model]
}

object ModelResolver {

  val MODEL_RESOLVER_IMPL_KEY = "rikai.sql.model_resolver.impl"

  private var resolver: Option[ModelResolver] = None

  def get(session: SparkSession): ModelResolver = {
    synchronized(
      if (resolver.isEmpty) {
        resolver = Some(Class
          .forName(session.conf.get(MODEL_RESOLVER_IMPL_KEY))
          .getDeclaredConstructor()
          .newInstance()
          .asInstanceOf[ModelResolver]
        )
      }
    )
    resolver.get
  }
}
