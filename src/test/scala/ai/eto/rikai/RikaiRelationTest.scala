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

import java.io.File
import java.nio.file.Files

import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class RikaiRelationTest extends AnyFunSuite {

  lazy val spark = SparkSession.builder().master("local[1]").getOrCreate()
  import spark.implicits._

  spark.sparkContext.setLogLevel("WARN")

  test("Use rikai registered as the sink of spark") {
    val examples = Seq(
      ("123", "car"),
      ("123", "people"),
      ("246", "tree")
    ).toDF("id", "label")

    val testDir =
      new File(Files.createTempDirectory("rikai").toFile(), "dataset")

    examples.write.format("rikai").save(testDir.toString())

    val numParquetFileds =
      testDir.list().filter(_.endsWith(".parquet")).length
    assert(numParquetFileds > 0)

    val df = spark.read.parquet(testDir.toString())
    assert(df.intersectAll(examples).count == 3)
  }
}
