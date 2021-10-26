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

import ai.eto.rikai.sql.model.{Model, ModelNotFoundException, ModelSpec, Registry, SparkUDFModel}
import ai.eto.rikai.sql.spark.Python
import org.apache.logging.log4j.scala.Logging
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.mlflow.tracking.MlflowContext

import scala.collection.JavaConverters._
import scala.util.Random

/** MLflow-based Model [[Registry]].
  */
class MlflowRegistry(val conf: Map[String, String], session: SparkSession)
  extends Registry
    with Logging {

  private val pyClass: String =
    "rikai.spark.sql.codegen.mlflow_registry.MlflowRegistry"

  /** Resolve a [[Model]] from the specific URI.
    *
    * @param spec Model Spec to send to python.
    * @throws ModelNotFoundException if the model does not exist on the registry.
    * @return [[Model]] if found.
    */
  @throws[ModelNotFoundException]
  override def resolve(
                        spec: ModelSpec
                      ): Model = {
    val trackingUri = conf("rikai.sql.ml.registry.mlflow.tracking_uri")
    val mlflowContext = new MlflowContext(trackingUri)
    val mlflowClient = mlflowContext.getClient
    val modelName = if (spec.getUri.startsWith("mlflow:///")) spec.getUri.drop("mlflow:///".length) else spec.getUri
    val versions =
      mlflowClient.getLatestVersions(modelName, List("None", "Staging", "Production").asJava).asScala
    val newest = versions.maxBy(_.getVersion.toInt)
    val runId = newest.getRunId
    println(s"runId $runId")
    val run = mlflowClient.getRun(runId)
    val flavor = run.getData.getTagsList.asScala.find(_.getKey == "rikai.model.flavor").map(_.getValue)
    //    val schema = run.getData.getTagsList.asScala.find(_.getKey == "rikai.output.schema").map(_.getValue)
    println(s"get flavor from run $flavor")
    flavor match {
      case Some("spark") =>
          val artifacts = mlflowClient.downloadArtifacts(runId)
          println(s"artifacts: $artifacts")
          val transformer = PipelineModel.load(artifacts.toURI.toString + "/model/sparkml")
          println(s"tran tran tran ${transformer.getClass}")
          def func(frames : Iterable[DataFrame]) : Iterable[DataFrame] = {
            frames.map(transformer.transform)
          }
          //replace with generated udf
          val funcName = modelName + Random.nextInt(0xffff)
          session.udf.register(funcName, func _)
          new SparkUDFModel(modelName,spec.getUri,funcName,Some("spark"))
      case _ =>
        Python.resolve(pyClass, spec)
    }
  }
}
