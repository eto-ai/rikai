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

package ai.eto.rikai.sql.spark.expressions

import ai.eto.rikai.SparkTestSession
import org.apache.spark.sql.rikai.Box2d
import org.scalatest.funsuite.AnyFunSuite

class SqlFunctionTest extends AnyFunSuite with SparkTestSession {
  import spark.implicits._

  test("test box area function") {

    Seq((1, new Box2d(1, 2, 3, 4)))
      .toDF("id", "box")
      .createOrReplaceTempView("boxes")
    val df = spark.sql("SELECT *, area(box) FROM boxes")
    df.show()
  }

}
