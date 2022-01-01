/*
 * Copyright 2022 Rikai authors
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

package ai.eto.rikai.sql.spark

import ai.eto.rikai.sql.model.Catalog
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.{BeforeAndAfterAll, Suite}

class MLPredictTest extends AnyFunSuite with BeforeAndAfterAll {

  this: Suite =>
  lazy val spark: SparkSession = SparkSession.builder
    .config(
      "spark.sql.extensions",
      "ai.eto.rikai.sql.spark.RikaiSparkSessionExtensions"
    )
    .config(
      Catalog.SQL_ML_CATALOG_IMPL_KEY,
      Catalog.SQL_ML_CATALOG_IMPL_DEFAULT
    )
    .config("spark.port.maxRetries", 128)
    .master("local[2]")
    .getOrCreate

  spark.sparkContext.setLogLevel("WARN")

  private def downloadResnet(): Unit = {
    Python.execute("""import torch;
        |import torchvision;
        |resnet = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        |    pretrained=True, progress=False);
        |torch.save(resnet, "/tmp/resnet.pt")""".stripMargin)
  }

  override def beforeAll(): Unit = {
    downloadResnet()
  }

  test("Use simple fs model") {
    spark.sql(
      "CREATE MODEL resnet USING '/tmp/resnet.pt'"
    )
    val df = spark.sql("SELECT ML_PREDICT(resnet, 1) as s")
    df.show()
    assert(df.exceptAll(spark.sql("SELECT 2 as s")).isEmpty)
  }
}
