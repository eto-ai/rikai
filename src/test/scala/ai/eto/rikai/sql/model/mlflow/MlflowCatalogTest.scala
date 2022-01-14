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

package ai.eto.rikai.sql.model.mlflow

import ai.eto.rikai.sql.model.mlflow.MlflowCatalog.ARTIFACT_PATH_KEY
import ai.eto.rikai.sql.spark.Python
import org.apache.spark.sql.rikai.{Box2dType, Image}
import org.apache.spark.sql.types._
import org.mlflow.api.proto.ModelRegistry.{CreateModelVersion, ModelVersionTag}
import org.mlflow.api.proto.Service.RunInfo
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite

import scala.util.Random

class MlflowCatalogTest
    extends AnyFunSuite
    with SparkSessionWithMlflow
    with BeforeAndAfterEach {

  import spark.implicits._

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
    val modelName = "testModel" + Random.nextInt(Int.MaxValue)
    mlflowClient.createModel(modelName)

    //Seems we can not really try a model in Java APi
    val builder = CreateModelVersion
      .newBuilder()
      .setName(modelName)
      .setSource("fake_uri")
      .setRunId(run.getRunId)
      .setRunLink(run.getArtifactUri)
    builder.addTags(
      ModelVersionTag
        .newBuilder()
        .setKey(ARTIFACT_PATH_KEY)
        .setValue("fake_uri")
        .build()
    )

    mlflowClient.client.sendPost(
      "model-versions/create",
      MlflowClientExt.jsonify(builder)
    )

    val models = spark.sql("SHOW MODELS")
    models.show()
  }

  test("test running a model registered model") {
    val script = getClass.getResource("/create_models.py").getPath
    Python.run(
      Seq(
        script,
        "--mlflow-uri",
        testMlflowTrackingUri,
        "--run-id",
        run.getRunId
      )
    )

    val modelsDf = spark.sql("SHOW MODELS")
    assert(
      modelsDf
        .exceptAll(
          Seq(
            ("resnet", "pytorch", "mlflow:/resnet", ""),
            ("ssd", "pytorch", "mlflow:/ssd", "")
          ).toDF("name", "flavor", "uri", "options")
        )
        .isEmpty
    )
    modelsDf.show()

    spark
      .createDataFrame(
        Seq(
          (1, new Image(getClass.getResource("/000000304150.jpg").getPath)),
          (2, new Image(getClass.getResource("/000000419650.jpg").getPath))
        )
      )
      .toDF("image_id", "image")
      .createOrReplaceTempView("images")

    val df =
      spark.sql("SELECT image_id, ML_PREDICT(ssd, image) as det FROM images")
    assert(
      df.schema == StructType(
        Seq(
          StructField("image_id", IntegerType, nullable = false),
          StructField(
            "det",
            ArrayType(
              StructType(
                Seq(
                  StructField("box", Box2dType, nullable = true),
                  StructField("score", FloatType, nullable = true),
                  StructField("label_id", IntegerType, nullable = true)
                )
              )
            ),
            nullable = true
          )
        )
      )
    )
    assert(
      spark
        .sql("""
      SELECT * FROM (
        SELECT size(ML_PREDICT(ssd, image)) as num_objects FROM images
      ) WHERE num_objects > 0
    """).count() == 2
    )
  }
}
