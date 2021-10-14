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
import org.scalactic.{Equality, TolerantNumerics}
import org.scalatest.funsuite.AnyFunSuite

class Box2dTest extends AnyFunSuite with SparkTestSession {

  implicit val doubleEq: Equality[Double] =
    TolerantNumerics.tolerantDoubleEquality(1e-4f)

  test("overlaps and intersect") {
    val box1 = new Box2d(0, 0, 20, 20)
    val box2 = new Box2d(10, 10, 30, 30)

    assert(box1.iou(box2) === 1.0 / 7)
    assert(box1 overlaps box2)

    val interBox = box1 & box2
    assert(interBox.isDefined)
    assert(interBox.map(b => b.area).getOrElse(0) == 100)

    val big = new Box2d(0, 0, 100, 100)
    assert((box2 & big).isDefined)
    assert((box2 & big).contains(box2))
  }

  test("not overlap") {
    val box1 = new Box2d(0, 0, 10, 10)
    val box2 = new Box2d(10, 10, 20, 20)
    assert(!(box1 overlaps box2))
    assert(!(box2 overlaps box1))
    assert((box1 & box2).isEmpty)

    val box3 = new Box2d(5, 10, 15, 20)
    val box4 = new Box2d(5, 0, 15, 10)
    assert(!box3.overlaps(box4))
  }
}
