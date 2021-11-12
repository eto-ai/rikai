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
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterEach, Suite}

trait SparkSessionWithMlflow extends BeforeAndAfterEach {

  this: Suite =>
  lazy val spark: SparkSession = SparkSession.builder
    .config(
      "spark.sql.extensions",
      "ai.eto.rikai.sql.spark.RikaiSparkSessionExtensions"
    )
    .config(
      Catalog.SQL_ML_CATALOG_IMPL_KEY,
      MlflowCatalog.SQL_ML_CATALOG_IMPL_MLFLOW
    )
    .config(
      MlflowCatalog.TrackingUriKey,
      sys.env.getOrElse("TEST_MLFLOW_TRACKING_URI", ""),
    )
    .config(
      Registry.REGISTRY_IMPL_PREFIX + "test.impl",
      "ai.eto.rikai.sql.model.testing.TestRegistry"
    )
    .config("spark.port.maxRetries", 128)
    .master("local[*]")
    .getOrCreate

  spark.sparkContext.setLogLevel("WARN")

  val testMlflowTrackingUri: String = sys.env.getOrElse("TEST_MLFLOW_TRACKING_URI", "")

}