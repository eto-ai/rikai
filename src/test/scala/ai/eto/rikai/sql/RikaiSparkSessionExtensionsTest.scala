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

import ai.eto.rikai.SparkTestSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}
import org.scalatest.funsuite.AnyFunSuite

class RikaiSparkSessionExtensionsTest
    extends AnyFunSuite
    with SparkTestSession {

  import spark.implicits._

  def assertDfEqual(actual: DataFrame, expected: DataFrame): Unit = {
    assert(actual.count() == expected.count())
    assert(actual.exceptAll(expected).isEmpty)
  }

  test("Test parse ML_PREDICT expression") {
    spark.udf.register("foo", (s: Int) => s + 2)

    val df = Seq.range(1, 10).toDF("id")
    df.createTempView("df")

    val scores =
      spark.sql("SELECT id, ML_PREDICT(model.`//foo`, id) AS score FROM df")
    // scores.show()

    val plus_two = udf((v: Int) => v + 2)
    val expected = df.withColumn("score", plus_two(col("id")))
    assertDfEqual(scores, expected)
  }

  test("Test parse ML_PREDICT on multiple columns") {
    spark.udf.register("multi_col", (a: Int, b: Int) => a * b)

    val df = Seq((1, 2), (3, 4), (10, 20), (17, 18)).toDF("a", "b")
    df.createTempView("multi_df")

    val predicted = spark.sql(
      "SELECT a + b as s, ML_PREDICT(model.`multi_col`, a, b) AS c FROM multi_df"
    )
//    predicted.show()

    val expected = Seq((3, 2), (7, 12), (30, 200), (35, 306)).toDF("s", "c")
    assertDfEqual(predicted, expected)
  }

  test("ML_PREDICT uses model catalog") {
    spark.udf.register("model_foo", (a: Int, b: Int) => a * b)

    val df = Seq((1, 2), (3, 4), (10, 20), (17, 18)).toDF("a", "b")
    df.createTempView("create_model_data")

    spark.sql("CREATE MODEL model_foo USING 'model.path_to_somewhere'").count()
    df.show()
    val actual = spark.sql(
      "SELECT a + b as s, ML_PREDICT(model_foo, a, b) AS c FROM create_model_data"
    )
    actual.show()
    val expected = Seq((3, 2), (7, 12), (30, 200), (35, 306)).toDF("s", "c")
    assertDfEqual(actual, expected)
  }

  test("Show Models") {
    spark.sql("CREATE MODEL model_foo USING 'model.path_to_somewhere'")
    val models = spark.sql("SHOW MODELS")
    val expected = Seq(("model_foo", "model.path_to_somewhere", "{}")).toDF()
    assertDfEqual(models, expected)
  }
}
