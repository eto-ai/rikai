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

import org.scalatest.funsuite.AnyFunSuite

import java.io.File
import java.nio.file.Files

class RikaiRelationTest extends AnyFunSuite with SparkTestSession {

  import spark.implicits._

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
  }

  test("Use rikai reader and writer") {
    import ai.eto.rikai._
    examples.write.rikai(testDir.toString)

    val numParquetFiles =
      testDir.list().count(_.endsWith(".parquet"))
    assert(numParquetFiles > 0)

    val df = spark.read.rikai(testDir.toString)
    assert(df.intersectAll(examples).count == 3)
  }

  test("Use partitions") {
    examples.write.partitionBy("label").rikai(testDir.toString)

    val partitions =
      Set(testDir.list().toSeq.filter(_.startsWith("label=")): _*)

    assert(partitions == Set("label=car", "label=people", "label=tree"))
  }

  test("test default block size") {
    val options = new RikaiOptions(Map.empty)
    assert(options.blockSize == RikaiOptions.defaultBlockSize)
  }
}
