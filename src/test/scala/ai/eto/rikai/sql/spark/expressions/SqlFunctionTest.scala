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

package ai.eto.rikai.sql.spark.expressions

import ai.eto.rikai.SparkTestSession
import org.apache.spark.sql.rikai.Box2d
import org.scalactic.TolerantNumerics
import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.sql.{functions => F}

class SqlFunctionTest extends AnyFunSuite with SparkTestSession {
  import spark.implicits._
  implicit val doubleEquality = TolerantNumerics.tolerantDoubleEquality(0.01)

  test("test box area function") {

    Seq((1, new Box2d(1, 2, 3, 4)))
      .toDF("id", "box")
      .createOrReplaceTempView("boxes")
    val df = spark.sql("SELECT *, area(box) as area FROM boxes")
    assert(df.first().getAs[Double]("area") == 4)
  }

  test("test iou function") {
    Seq((1, new Box2d(0, 0, 20, 20), new Box2d(10, 10, 30, 30)))
      .toDF("id", "box1", "box2")
      .createOrReplaceTempView("boxes")
    val df = spark.sql("SELECT *, iou(box1, box2) as iou FROM boxes")
    assert(df.first().getAs[Double]("iou") === 1.0 / 7)
  }

  test("codegen explore") {
    spark.conf.set("spark.sql.codegen.fallback", false)
    spark.conf.set("spark.sql.codegen.factoryMode", "CODEGEN_ONLY")

    spark.sql("select box2d(0, 0, 20, 20)").show()

    val q = spark
      .range(2)
      .select(
        F.col("id").as("id"),
        F.rand(10).as("xmin"),
        F.rand(10).as("ymin"),
        F.lit(100).as("xmax"),
        F.lit(100).as("ymax")
      )
      .selectExpr(
        "box2d(xmin, ymin, xmax, ymax) as box1",
        "box2d(xmin + 1, ymin + 1, xmax, ymax) as box2"
      )
      .selectExpr(
        "iou(box1, box2) as iou"
      )

    q.show(10, 100, true)
  }
}
