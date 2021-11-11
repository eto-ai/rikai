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

import ai.eto.rikai.sql.model.mlflow.MlflowCatalog.{
  ArtifactPathKey,
  ModelFlavorKey
}
import ai.eto.rikai.sql.model.{Catalog, Model, SparkUDFModel}
import com.google.protobuf.InvalidProtocolBufferException
import org.mlflow.api.proto.ModelRegistry.SearchRegisteredModels
import org.mlflow.tracking.{MlflowClient, MlflowClientException}
import org.mlflow_project.google.protobuf.Message.Builder
import org.mlflow_project.google.protobuf.util.JsonFormat

import scala.collection.JavaConverters._

/** Use MLflow as a persisted backend for Model Catalog
  */
class MlflowCatalog(val mlflowClient: MlflowClient) extends Catalog {

  /** Use default trackingUri to build Mlflow Catalog */
  def this() = {
    this(new MlflowClient())
    println("Hey, I was created")
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
    response.getRegisteredModelsList.asScala
      .map(model => {
        model.getLatestVersionsCount match {
          case 0 => None
          case _ => {
            val latestVersion = model.getLatestVersions(0)
            val tagsMap = latestVersion.getTagsList.asScala
              .map(t => t.getKey -> t.getValue)
              .toMap
            val name = model.getName
            if (tagsMap.contains(ArtifactPathKey)) {
              val flavor = tagsMap.getOrElse(ModelFlavorKey, "")
              Some(
                new SparkUDFModel(
                  name,
                  s"mlflow://$name",
                  "<anonymous>",
                  flavor
                )
              )
            } else {
              None
            }
          }
        }
      })
      .filter(m => m.isDefined)
      .map(m => m.get)
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

  val TrackingUriKey = "rikai.sql.ml.registry.mlflow.tracking_uri"

  val ArtifactPathKey = "rikai.model.artifact_path"
  val ModelFlavorKey = "rikai.model.flavor"
  val OutputSchemaKey = "rikai.output.schema"
  val SpecVersionKey = "rikai.spec.version"
  val PreProcessingKey = "rikai.transforms.pre"
  val PostProcessingKey = "rikai.transforms.post"

  val SQL_ML_CATALOG_IMPL_MLFLOW = "ai.eto.rikai.sql.model.mlflow.MlflowCatalog"

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
