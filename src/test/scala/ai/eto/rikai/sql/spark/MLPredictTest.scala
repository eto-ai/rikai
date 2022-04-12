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
import org.apache.spark.sql.rikai.{Box2dType, Image}
import org.apache.spark.sql.types._
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import java.nio.file.{Files, Path}

class MLPredictTest
    extends AnyFunSuite
    with SparkTestSession
    with BeforeAndAfterAll {

  lazy val resnetPath: Path = Files.createTempFile("resnet", ".pt")

  private def downloadResnet(): Unit = {
    Python.execute(f"""import torch;
        |import torchvision;
        |fasterrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        |    pretrained=True, progress=False);
        |torch.save(fasterrcnn, "${resnetPath}")""".stripMargin)
  }

  private val resnetSpecYaml: String = s"""
      |version: "1.0"
      |name: resnet
      |model:
      |  uri: ${resnetPath}
      |  flavor: pytorch
      |  type: fasterrcnn
      """.stripMargin

  override def beforeAll(): Unit = {
    super.beforeAll()
    downloadResnet()
  }

  override def afterAll(): Unit = {
    super.afterAll()
    Files.delete(resnetPath)
  }

  test("Use simple fs model") {
    val specYamlPath = Files.createTempFile("spec", ".yml")
    try {
      Files.write(specYamlPath, resnetSpecYaml.getBytes)
      spark.sql(
        s"CREATE MODEL resnet USING 'file://${specYamlPath}'"
      )

      spark
        .createDataFrame(
          Seq(
            (1, new Image(getClass.getResource("/000000304150.jpg").getPath)),
            (2, new Image(getClass.getResource("/000000419650.jpg").getPath))
          )
        )
        .toDF("image_id", "image")
        .createOrReplaceTempView("images")

      val df = spark.sql(
        """SELECT
          |ML_PREDICT(resnet, image) AS s FROM images""".stripMargin
      )
      df.cache()
      df.show()
      df.printSchema()
      assert(df.count() == 2)
      assert(
        df.schema == StructType(
          Seq(
            StructField(
              "s",
              ArrayType(
                StructType(
                  Seq(
                    StructField("box", Box2dType),
                    StructField("score", FloatType),
                    StructField("label_id", IntegerType),
                    StructField("label", StringType)
                  )
                )
              )
            )
          )
        )
      )
    } finally {
      Files.delete(specYamlPath)
    }
  }

  test("Test Column name") {
    val specYamlPath = Files.createTempFile("spec", ".yml")
    try {
      Files.write(specYamlPath, resnetSpecYaml.getBytes)
      spark.sql(
        s"CREATE MODEL resnet USING 'file://${specYamlPath}'"
      )

      spark
        .createDataFrame(
          Seq(
            (1, new Image(getClass.getResource("/000000304150.jpg").getPath))
          )
        )
        .toDF("image_id", "image")
        .createOrReplaceTempView("images")

      val df = spark.sql(
        """SELECT
          |ML_PREDICT(resnet, image), ML_PREDICT(resnet, image) AS pred
          |FROM images""".stripMargin
      )
      df.printSchema()
      df.show()
      assert(
        df.schema == StructType(
          Seq(
            StructField(
              "resnet",
              ArrayType(
                StructType(
                  Seq(
                    StructField("box", Box2dType),
                    StructField("score", FloatType),
                    StructField("label_id", IntegerType),
                      StructField("label", StringType)
                  )
                )
              )
            ),
            StructField(
              "pred",
              ArrayType(
                StructType(
                  Seq(
                    StructField("box", Box2dType),
                    StructField("score", FloatType),
                    StructField("label_id", IntegerType),
                    StructField("label", StringType)
                  )
                )
              )
            )
          )
        )
      )
    } finally {
      Files.delete(specYamlPath)
    }
  }
}
