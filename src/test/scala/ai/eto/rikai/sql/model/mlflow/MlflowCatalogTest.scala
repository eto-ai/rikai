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

import org.mlflow.api.proto.Service.RunInfo
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite

import scala.util.Random

class MlflowCatalogTest
    extends AnyFunSuite
    with SparkSessionWithMlflow
    with BeforeAndAfterEach {

  var run: RunInfo = null

  override def beforeEach(): Unit = {
    run = mlflowClient.client.createRun()
    super.beforeEach()
  }

  override def afterEach(): Unit = {
    mlflowClient.client.deleteRun(run.getRunId)
    clearModels()
    super.afterEach()
  }

  test("test list registered models") {
    mlflowClient.createModel("testModel" + Random.nextInt(Int.MaxValue))
    //TODO train a model version
    val models = spark.sql("SHOW MODELS")
    models.show()
  }
}
