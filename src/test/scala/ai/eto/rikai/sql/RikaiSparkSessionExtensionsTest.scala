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

package ai.eto.rikai.sql

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, udf}
import org.scalatest.funsuite.AnyFunSuite

class RikaiSparkSessionExtensionsTest extends AnyFunSuite {

  lazy val spark = SparkSession.builder
    .config(
      "spark.sql.extensions",
      "ai.eto.rikai.sql.RikaiSparkSessionExtensions"
    )
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

  test("Test parse ML_PREDICT expression") {
    spark.udf.register("foo_udf", (s: Int) => s + 2)

    val df = Seq.range(1, 10).toDF("id")
    df.show()
    df.createTempView("df")

    val scores =
      spark.sql("SELECT id, ML_PREDICT(model.`//foo`, id) AS score FROM df")
    scores.show()

    val plus_two = udf((v: Int) => v + 2)
    val expected = df.withColumn("score", plus_two(col("id")))
    assert(expected.count() == scores.count())
    assert(expected.exceptAll(scores).isEmpty)
  }
}
