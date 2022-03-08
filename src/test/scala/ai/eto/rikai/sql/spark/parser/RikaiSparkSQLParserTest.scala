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

package ai.eto.rikai.sql.spark.parser

import ai.eto.rikai.SparkTestSession
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.rikai.{Box2d, Image}
import org.apache.spark.sql.types._
import org.scalatest.funsuite.AnyFunSuite

class RikaiSparkSQLParserTest extends AnyFunSuite with SparkTestSession {

  import spark.implicits._

  val df = Seq.range(1, 10).toDF("id")
  df.createOrReplaceTempView("df")

  test("Test parse ML_PREDICT expression") {
    cancel("TODO: temporarily disabled")

    spark.udf.register("foo", (s: Int) => s + 2)

    val scores =
      spark.sql("SELECT id, ML_PREDICT('test://host/foo', id) AS score FROM df")

    val plus_two = udf((v: Int) => v + 2)
    val expected = df.withColumn("score", plus_two(col("id")))
    assertEqual(scores, expected)
  }

  test("Test parse ML_PREDICT with catalog") {
    spark.udf.register("bar", (s: Int) => s + 2)

    spark.sql("CREATE MODEL bar USING 'test://host/bar'").show()

    val scores =
      spark.sql("SELECT id, ML_PREDICT(bar, id) AS score FROM df")

    val plus_two = udf((v: Int) => v + 2)
    val expected = df.withColumn("score", plus_two(col("id")))
    assertEqual(scores, expected)
  }

  test("Test parse explain select 1") {
    val df = spark.sql("explain select 1")
    assert(df.count() === 1)
  }

  test("test to_struct on Image") {
    val images_df = Seq((new Image("s3://foo/bar"), 1)).toDF("image", "id")
    images_df.createOrReplaceTempView("images")
    val df = spark.sql("SELECT to_struct(image) as img FROM images")
    df.show()
    df.printSchema()
    assert(
      df.schema == StructType(
        Seq(
          StructField(
            "img",
            StructType(
              Seq(
                StructField("data", BinaryType),
                StructField("uri", StringType)
              )
            )
          )
        )
      )
    )
    assert(df.count() == 1)
    assert(df.select("img.uri").collect()(0) == Row("s3://foo/bar"))
  }

  test("test to_struct over box2d") {
    val boxes = Seq((new Box2d(1, 2, 3, 4), 1), (new Box2d(10, 20, 30, 40), 2))
      .toDF("box", "id")
    boxes.createOrReplaceTempView("boxes")

    val df =
      spark.sql(
        "SELECT to_struct(box) as bigbox FROM boxes WHERE to_struct(box).xmin > 3"
      )
    df.show()
    df.printSchema()
    assert(
      df.schema == StructType(
        Seq(
          StructField(
            "bigbox",
            StructType(
              Seq(
                StructField("xmin", DoubleType, nullable = false),
                StructField("ymin", DoubleType, nullable = false),
                StructField("xmax", DoubleType, nullable = false),
                StructField("ymax", DoubleType, nullable = false)
              )
            )
          )
        )
      )
    )
    assert(df.count() == 1)
    assert(df.collect()(0) == Row(Row(10.0, 20.0, 30.0, 40.0)))
  }
}
