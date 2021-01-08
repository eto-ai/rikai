/*
 * Copyright 2020 Rikai authors
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

package org.apache.spark.sql.rikai

import java.io.File
import java.nio.file.Files
import scala.reflect.io.Directory

import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession

import org.scalatest.funsuite.AnyFunSuite

class LabelTest extends AnyFunSuite {

  lazy val spark = SparkSession.builder().master("local[2]").getOrCreate()

  import spark.implicits._

  spark.sparkContext.setLogLevel("ERROR")

  val label = udf { v: String => new Label(v) }

  test("test serialize label") {
    val testDir =
      new File(Files.createTempDirectory("rikai").toFile(), "dataset")

    var df = Seq(("a"), ("b"), ("c")).toDF()
    df = df.withColumn("value", label(col("value")))
    df.write.format("rikai").save(testDir.toString())
    df.show()

    val actualDf = spark.read.format("rikai").load(testDir.toString())
    assert(df.count() == actualDf.count())
    assert(df.collect() sameElements actualDf.collect())
    actualDf.show()

    new Directory(testDir).deleteRecursively()
  }

  test("test label in struct") {
    val testDir =
      new File(Files.createTempDirectory("rikai").toFile(), "dataset")

    val df =
      Seq((3.5, new Label("a")), (2.2, new Label("b")), (3.5, new Label("c")))
        .toDF("id", "label")
    df.write.mode("overwrite").format("rikai").save(testDir.toString())
    df.show()

    val actualDf =
      spark.read.format("rikai").load(testDir.toString())
    actualDf.printSchema()
    assert(df.count() == actualDf.count())
    assert(df.exceptAll(actualDf).isEmpty)

    new Directory(testDir).deleteRecursively()
  }
}
