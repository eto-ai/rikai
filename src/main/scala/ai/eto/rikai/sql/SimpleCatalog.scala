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

package ai.eto.rikai.sql

case class SimpleCatalog() extends Catalog {

  private var models: Map[String, Model] = Map.empty

  /**
    * Create a ML Model that can be used in SQL ML in the current database.
    */
  override def createModel(model: Model): Model = {
    models += (model.name -> model)
    model
  }

  /**
    * Return a list of models available for all Sessions
    */
  override def listModels(): Seq[Model] = {
    models.values.toSeq
  }

  /**
    * Check a model with the specified name exists.
    *
    * @param name is the name of the model.
    */
  override def modelExists(name: String): Boolean = models.contains(name)

  /**
    * Get the model with a specific name.
    *
    * @param name is a qualified name pointed to a Model.
    * @return the model
    */
  override def getModel(name: String): Option[Model] = {
    models get name
  }

  /**
    * Drops a model with a specific name
    *
    * @param name the model name
    * @return true of the model is dropped successfully. False otherwise.
    */
  override def dropModel(name: String): Boolean = {
    val contains = models.contains(name)
    models -= name
    contains
  }
}
