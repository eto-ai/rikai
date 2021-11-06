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

package ai.eto.rikai.sql.model.mlflow

import ai.eto.rikai.sql.model.{Catalog, Model}
import org.mlflow.tracking.MlflowClient
import org.mlflow.api.proto.Service.ViewType;

import java.util

/** Use MLflow as a persisted backend for Model Catalog */
class MlflowCatalog(val mlflowClient: MlflowClient) extends Catalog {

  def this() = {
    this(new MlflowClient())
  }

  def this(trackingUri: String) = {
    this(new MlflowClient(trackingUri))
  }

  /** Create a ML Model that can be used in SQL ML in the current database.
    */
  override def createModel(model: Model): Model =
    throw new NotImplementedError()

  /** Return a list of models available for all Sessions
    */
  override def listModels(): Seq[Model] = {
    val modelFilter = "tag.`rikai.output.schema` != \"\"";
    val defaultExperiments = new util.ArrayList[String](1);
    defaultExperiments.add("0")
    val results = mlflowClient.searchRuns(
      defaultExperiments,
      modelFilter,
      ViewType.ACTIVE_ONLY,
      100
    )
    println(results)
    Seq()
  }

  /** Check a model with the specified name exists.
    *
    * @param name is the name of the model.
    */
  override def modelExists(name: String): Boolean = ???

  /** Get the model with a specific name.
    *
    * @param name is a qualified name pointed to a Model.
    * @return the model
    */
  override def getModel(name: String): Option[Model] = ???

  /** Drops a model with a specific name
    *
    * @param name the model name
    * @return true of the model is dropped successfully. False otherwise.
    */
  override def dropModel(name: String): Boolean =
    throw new NotImplementedError()
}
