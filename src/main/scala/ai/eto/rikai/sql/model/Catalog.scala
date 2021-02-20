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

/**
  * Catalog for SQL ML.
  */
trait Catalog {

  /**
    * Create a ML Model that can be used in SQL ML in the current database.
    */
  def createModel(model: Model): Model

  /**
    * Return a list of models available for all Sessions
    */
  def listModels(): Seq[Model]

  /**
    * Check a model with the specified name exists.
    *
    * @param name is the name of the model.
    */
  def modelExists(name: String): Boolean

  /**
    * Get the model with a specific name.
    *
    * @param name is a qualified name pointed to a Model.
    * @return the model
    */
  def getModel(name: String): Option[Model]

  /**
    * Drops a model with a specific name
    *
    * @param name the model name
    * @return true of the model is dropped successfully. False otherwise.
    */
  def dropModel(name: String): Boolean
}

object Catalog {

  val SQL_ML_CATALOG_IMPL_KEY = "rikai.sql.ml.catalog.impl"
  val SQL_ML_CATALOG_IMPL_DEFAULT = "ai.eto.rikai.sql.model.SimpleCatalog"

  /** A Catalog for local testing. */
  private[rikai] def testing: SimpleCatalog =
    getOrCreate(SQL_ML_CATALOG_IMPL_DEFAULT).asInstanceOf[SimpleCatalog]

  private var catalog: Catalog = null

  def getOrCreate(className: String): Catalog = {
    if (catalog == null) {
      catalog = Class
        .forName(className)
        .getDeclaredConstructor()
        .newInstance()
        .asInstanceOf[Catalog]
    }
    catalog
  }
}
