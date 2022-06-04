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

class VideoTest extends AnyFunSuite with SparkTestSession {
  import spark.implicits._

  test("test video serde") {
    val testDir =
      new File(Files.createTempDirectory("rikai").toFile, "video_dataset")

    val df = Seq(
      (
        1,
        new Segment(10, 253),
        new YouTubeVideo("vid"),
        new VideoStream("uri")
      ),
      (
        2,
        new Segment(57934, 5812952),
        new YouTubeVideo("vid"),
        new VideoStream("uri")
      )
    ).toDF()

    df.write.format("rikai").mode(SaveMode.Overwrite).save(testDir.toString())

    val actualDf = spark.read.format("rikai").load(testDir.toString())
    assert(df.count() == actualDf.count())
    assert(df.exceptAll(actualDf).isEmpty)

    new Directory(testDir).deleteRecursively()
  }
}
