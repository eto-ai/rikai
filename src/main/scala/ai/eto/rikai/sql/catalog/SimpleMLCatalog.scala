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

/**
  * An Simple ML-Catalog
  */
class SimpleMLCatalog extends MLCatalog {

  private var models: Map[String, Model] = Map.empty

  /**
    * Create a ML Model that can be used in SQL ML in the current database.
    *
    * @param model The model to create.
    */
  override def createModel(model: Model): Model = {
    models += (model.name -> model)
    model
  }

  /**
    * Return a list of models registered in the current database.
    */
  override def listModels(): Seq[Model] = {
    models.values.toSeq
  }

  /** Drop the model, specified by the name. */
  override def dropModel(name: String): Boolean = {
    val contains = models.contains(name)
    models -= name
    contains
  }

  override def getModel(name: String): Option[Model] = {
    models get name
  }
}
