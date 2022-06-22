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

package org.apache.spark.sql.rikai

import ai.eto.rikai.SparkTestSession
import org.apache.spark.sql.SaveMode
import org.scalatest.funsuite.AnyFunSuite

import java.io.File
import java.nio.file.Files
import scala.reflect.io.Directory

class PointTest extends AnyFunSuite with SparkTestSession {

  import spark.implicits._

  test("test point equality") {
    assert(new Point(1, 2, 3) == new Point(1, 2, 3))
    assert(new Point(1, 2, 3) != new Point(10, 20, 30))
  }

  test("test serialize points") {
    val testDir =
      new File(Files.createTempDirectory("rikai").toFile(), "dataset")

    val df = Seq((1, new Point(1, 2, 3))).toDF()

    df.write.format("rikai").mode(SaveMode.Overwrite).save(testDir.toString())
    df.show()

    val actualDf = spark.read.format("rikai").load(testDir.toString())
    assert(df.count() == actualDf.count())
    actualDf.show()
    assert(df.collect() sameElements actualDf.collect())

    new Directory(testDir).deleteRecursively()
  }
}
