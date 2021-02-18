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

import org.apache.spark.sql.{Dataset, SparkSession}

/**
  * Catalog for SQL ML.
  *
  * Similar to ``org.apache.spark.sql.catalog.Catalog``, this abstract class will be
  * used in Java for extension as well.
  */
abstract class MLCatalog {

  /**
    * Create a ML Model that can be used in SQL ML in the current database.
    */
  def createModel(model: Model): Model

  /**
    * Return a list of models registered in the current database.
    */
  def listModels(): Dataset[Model]

  /** Drop the model, specified by the name. */
  def dropModel(name: String): Boolean

  /**
    * Get [[Model]] by name.
    *
    * @param name model name
    *
    * @return the model specified by the name.
    */
  def getModel(name: String): Option[Model]
}

object MLCatalog {

  private var catalog: MLCatalog = new SimpleMLCatalog()

  /**
    * Get the singleton of [[MLCatalog]].
    *
    * @param session
    * @return
    */
  def get(session: SparkSession) = catalog

  /**
   * Clear the catalog. Test use only.
   */
  def clear(session: SparkSession): Unit = {
    catalog = new SimpleMLCatalog()
  }
}
