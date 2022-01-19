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

import ai.eto.rikai.sql.model.{Catalog, Registry}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.SparkSession
import org.mlflow.api.proto.Service.RunInfo
import org.scalatest.{BeforeAndAfterAll, BeforeAndAfterEach, Suite}

import scala.collection.JavaConverters._

trait SparkSessionWithMlflow
    extends LazyLogging
    with BeforeAndAfterAll
    with BeforeAndAfterEach {
  this: Suite =>

  val testMlflowTrackingUri: String =
    sys.env.getOrElse("TEST_MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

  var run: RunInfo = _

  lazy val spark: SparkSession = {
    SparkSession.builder
      .config(
        "spark.sql.extensions",
        "ai.eto.rikai.sql.spark.RikaiSparkSessionExtensions"
      )
      .config(
        Catalog.SQL_ML_CATALOG_IMPL_KEY,
        MlflowCatalog.SQL_ML_CATALOG_IMPL_MLFLOW
      )
      .config(
        MlflowCatalog.TRACKING_URI_KEY, {
          testMlflowTrackingUri
        }
      )
      .config(
        Registry.REGISTRY_IMPL_PREFIX + "test.impl",
        "ai.eto.rikai.sql.model.testing.TestRegistry"
      )
      .config("spark.port.maxRetries", 128)
      .master("local[*]")
      .getOrCreate
  }

  spark.sparkContext.setLogLevel("WARN")

  lazy val mlflowClient = new MlflowClientExt(testMlflowTrackingUri)

  override def beforeEach(): Unit = {
    super.beforeEach()
    run = mlflowClient.client.createRun()
  }

  override def afterEach(): Unit = {
    mlflowClient.client.deleteRun(run.getRunId)
    clearModels()
    dropTables()
    super.afterEach()
  }

  private def clearModels(): Unit = {
    mlflowClient
      .searchRegisteredModels()
      .getRegisteredModelsList
      .asScala
      //The MLFlow client does not support delete a model
      .foreach(m => mlflowClient.deleteModel(m.getName))
  }

  private def dropTables(): Unit = {
    spark.catalog
      .listTables()
      .collect()
      .foreach(t => spark.sql(s"DROP TABLE ${t.name}"))
  }

  override protected def afterAll(): Unit = {
    spark.close()
  }
}
