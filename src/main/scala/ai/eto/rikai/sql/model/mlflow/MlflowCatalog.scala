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
import com.google.protobuf.InvalidProtocolBufferException
import org.mlflow.api.proto.ModelRegistry.SearchRegisteredModels
import org.mlflow.tracking.{MlflowClient, MlflowClientException}
import org.mlflow_project.google.protobuf.Message.Builder
import org.mlflow_project.google.protobuf.util.JsonFormat

/** Use MLflow as a persisted backend for Model Catalog
  */
class MlflowCatalog(val mlflowClient: MlflowClient) extends Catalog {

  /** Use default trackingUri to build Mlflow Catalog */
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

  /** Return a list of models available for all Sessions */
  override def listModels(): Seq[Model] = {
    val response = searchRegisteredModels()
    println(response)
    response.getRegisteredModelsList()
    Seq()
  }

  /** Query models from mlflow
    *
    * TODO: contribute this back to mlflow.
    */
  private def searchRegisteredModels(): SearchRegisteredModels.Response = {
    val payload = mlflowClient.sendGet("registered-models/search")
    val builder = SearchRegisteredModels.Response.newBuilder()
    MlflowCatalog.merge(payload, builder)
    builder.build()
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

object MlflowCatalog {

  /** Merge json payload to the protobuf builder. */
  private def merge(
      jsonPayload: String,
      builder: Builder
  ) = {
    try {
      JsonFormat.parser.ignoringUnknownFields.merge(jsonPayload, builder)
    } catch {
      case e: InvalidProtocolBufferException =>
        throw new MlflowClientException(
          "Failed to serialize json " + jsonPayload + " into " + builder,
          e
        )
    }
  }
}
