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
  MODEL_FLAVOR_KEY,
  TRACKING_URI_KEY
}
import ai.eto.rikai.sql.model.{Catalog, Model, SparkUDFModel}
import org.apache.spark.SparkConf

import scala.collection.JavaConverters._

/** Use MLflow as a persisted backend for Model Catalog
  */
class MlflowCatalog(val conf: SparkConf) extends Catalog {

  private val mlflowClient = new MlflowClientExt(conf.get(TRACKING_URI_KEY))

  /** Create a ML Model that can be used in SQL ML in the current database.
    */
  override def createModel(model: Model): Model = {
    val name = mlflowClient.createModel(
      model.name,
      Map() ++ model.flavor.map(MODEL_FLAVOR_KEY -> _)
    )
    new SparkUDFModel(
      name,
      s"mlflow://$name",
      "<anonymous>",
      model.flavor
    )
  }

  /** Return a list of models available for all Sessions */
  override def listModels(): Seq[Model] = {
    val response = mlflowClient.searchRegisteredModels()
    response.getRegisteredModelsList.asScala
      .collect {
        case model if model.getLatestVersionsCount != 0 =>
          val latestVersion = model.getLatestVersions(0)
          val tagsMap = latestVersion.getTagsList.asScala
            .map(t => t.getKey -> t.getValue)
            .toMap
          val name = model.getName
          if (
            true
            //              TODO enable this after https://github.com/eto-ai/rikai/pull/351 finished
            //              && tagsMap.contains(ArtifactPathKey)
          ) {
            val flavor = tagsMap.getOrElse(MODEL_FLAVOR_KEY, "")
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
      .filter(m => m.isDefined)
      .map(m => m.get)
  }

  /** Check a model with the specified name exists.
    *
    * @param name is the name of the model.
    */
  override def modelExists(name: String): Boolean = {
    mlflowClient.getModel(name).nonEmpty
  }

  /** Get the model with a specific name.
    *
    * @param name is a qualified name pointed to a Model.
    * @return the model
    */
  override def getModel(name: String): Option[Model] = {
    mlflowClient.getModel(name).map { modelVersion =>
      val tagsMap = modelVersion.getTagsList.asScala
        .map(t => t.getKey -> t.getValue)
        .toMap
      new SparkUDFModel(
        name,
        s"mlflow://$name",
        "<anonymous>",
        tagsMap.get(MODEL_FLAVOR_KEY)
      )
    }
  }

  /** Drops a model with a specific name
    *
    * @param name the model name
    * @return true of the model is dropped successfully. False otherwise.
    */
  override def dropModel(name: String): Boolean = {
    try {
      mlflowClient.deleteModel(name)
      true
    } catch {
      case e: Exception =>
        e.printStackTrace()
        false
    }
  }
}

object MlflowCatalog {

  val TRACKING_URI_KEY = "rikai.sql.ml.registry.mlflow.tracking_uri"

  val ARTIFACT_PATH_KEY = "rikai.model.artifact_path"
  val MODEL_FLAVOR_KEY = "rikai.model.flavor"
  val OUTPUT_SCHEMA_KEY = "rikai.output.schema"
  val SPEC_VERSION_KEY = "rikai.spec.version"
  val PRE_PROCESSING_KEY = "rikai.transforms.pre"
  val POST_PROCESSING_KEY = "rikai.transforms.post"

  val SQL_ML_CATALOG_IMPL_MLFLOW = "ai.eto.rikai.sql.model.mlflow.MlflowCatalog"
}
