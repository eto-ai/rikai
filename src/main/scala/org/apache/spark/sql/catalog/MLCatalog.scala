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

package org.apache.spark.sql.catalog

import org.apache.spark.sql.Dataset

import java.net.URI


/**
  * Catalog for SQL ML.
  */
trait MLCatalog {

  /**
    * Create a ML Model that can be used in SQL ML in the current database.
    */
  def createModel(uri: URI) : Model

  /**
    * Return a list of models registered in the current database.
    */
  def listModels() : Dataset[Model]

  /** Drop the model, specified by the name. */
  def dropModel(name: String) : Boolean

  def getModel(name: String) : Option[Model]
}
