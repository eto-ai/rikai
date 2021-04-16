/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.eto.rikai

import org.apache.commons.io.FileUtils
import org.scalatest.funsuite.AnyFunSuite

import java.io.File
import java.nio.file.Files

class RikaiDataSourceV2Test extends AnyFunSuite with SparkTestSession {

  import spark.implicits._
  import ai.eto.rikai._

  private val examples = Seq(
    ("123", "car"),
    ("123", "people"),
    ("246", "tree")
  ).toDF("id", "label")

  private val testDir =
    new File(Files.createTempDirectory("rikai").toFile, "dataset")

  test("Use rikai registered as the sink of spark") {
    examples.write.format("rikai").save(testDir.toString)

    val numParquetFileds =
      testDir.list().count(_.endsWith(".parquet"))
    assert(numParquetFileds > 0)

    val df = spark.read.parquet(testDir.toString)
    assert(df.intersectAll(examples).count == 3)

    FileUtils.deleteDirectory(testDir)
  }

  test("Use rikai reader and writer") {
    examples.write.rikai(testDir.toString)

    val numParquetFileds =
      testDir.list().count(_.endsWith(".parquet"))
    assert(numParquetFileds > 0)

    val df = spark.read.rikai(testDir.toString)
    assert(df.intersectAll(examples).count == 3)

    FileUtils.deleteDirectory(testDir)
  }

  test("Write and read with predicate") {
    examples.write.rikai(testDir.toString)
    val df = spark.read.rikai(testDir.toString)
    df.createOrReplaceTempView("df")
    val query = spark.sql("select * from df where label = 'car'")
    assert(query.count() == 1)

    FileUtils.deleteDirectory(testDir)
  }
}
