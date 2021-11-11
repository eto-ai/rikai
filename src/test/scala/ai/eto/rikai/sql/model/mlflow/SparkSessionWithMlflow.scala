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
import org.mlflow.tracking.MlflowClient
import org.scalatest.{BeforeAndAfterEach, Suite}

import java.net.ServerSocket
import scala.sys.process._

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
      Registry.REGISTRY_IMPL_PREFIX + "test.impl",
      "ai.eto.rikai.sql.model.testing.TestRegistry"
    )
    .config("spark.port.maxRetries", 128)
    .master("local[*]")
    .getOrCreate

  spark.sparkContext.setLogLevel("WARN")

  lazy val mlflowClient: MlflowClient = startMlflowServer()

  private var process: Option[Process] = None

  def destroy = {
    process.foreach(p => p.destroy())
    process = None
  }

  /** Start python-based mlflow server in a process, and returns the connected MlflowClient */
  private def startMlflowServer(): MlflowClient = {
    val port = getFreePort
    process = Some(
      Seq("mlflow", "server", "--host", "127.0.0.1", "--port", s"$port").run()
    )
    new MlflowClient(s"http://127.0.0.1:$port")
  }

  private def getFreePort: Int = {
    // *nix systems rarely reuse recently allocated ports,
    // so we allocate one and then release it.
    val sock = new ServerSocket(0)
    val port = sock.getLocalPort
    sock.close()
    port
  }
}
