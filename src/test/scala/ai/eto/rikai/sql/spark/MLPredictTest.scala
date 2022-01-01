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

import ai.eto.rikai.SparkTestSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import java.nio.file.{Files, Path}

class MLPredictTest
    extends AnyFunSuite
    with SparkTestSession
    with BeforeAndAfterAll {

  import spark.implicits._

  lazy val resnetPath: Path = Files.createTempFile("resnet", ".pt")

  private def downloadResnet(): Unit = {
    Python.execute(f"""import torch;
        |import torchvision;
        |resnet = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        |    pretrained=True, progress=False);
        |torch.save(resnet, "${resnetPath}")""".stripMargin)
  }

  private val resnetSpecYaml: String = s"""
      |version: "1.0"
      |name: resnet
      |model:
      |  uri: ${resnetPath}
      |  flavor: pytorch
      |schema: STRUCT<boxes:ARRAY<ARRAY<float>>, scores:ARRAY<float>, label_ids:ARRAY<int>>
      |transforms:
      |  pre: rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.pre_processing
      |  post: rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.post_processing""".stripMargin

  override def beforeAll(): Unit = {
    super.beforeAll()
    downloadResnet()
  }

  test("Use simple fs model") {
    val specYamlPath = Files.createTempFile("spec", ".yml")
    try {
      Files.writeString(specYamlPath, resnetSpecYaml)
      spark.sql(
        s"CREATE MODEL resnet USING 'file://${specYamlPath}'"
      )
      Seq(
        getClass.getResource("000000304150.jpg").getPath,
        getClass.getResource("000000419650.jpg").getPath
      ).toDF("image_uri").createOrReplaceTempView("images")

      val df = spark.sql(
        """SELECT
          |ML_PREDICT(resnet, image_uri) AS s FROM images""".stripMargin
      )
      df.show()
      df.printSchema()
    } finally {
      Files.delete(specYamlPath)
    }
  }
}
