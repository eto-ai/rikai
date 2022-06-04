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

class MaskTest extends AnyFunSuite with SparkTestSession {

  import spark.implicits._

  test("test create mask data") {
    val testDir =
      new File(Files.createTempDirectory("rikai").toFile, "dataset")

    val df =
      Seq(
        (1, Mask.fromRLE(Array(1, 2, 3), 10, 5)),
        (2, Mask.fromPolygon(Array(Array(1, 1, 5, 5, 10, 10)), 5, 20))
      ).toDF("id", "segmentation")

    df.write.mode(SaveMode.Overwrite).format("rikai").save(testDir.toString)

    val actualDf = spark.read.load(testDir.toString)
    assert(df.count() == actualDf.count())
    assert(df.exceptAll(actualDf).isEmpty)
    df.show()

    val polygon = actualDf.filter("id = 2").first()
    assert(polygon.getAs[Mask]("segmentation").width == 5);
    assert(polygon.getAs[Mask]("segmentation").height == 20);
  }
}
