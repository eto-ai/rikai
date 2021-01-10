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

import org.apache.spark.sql.types._
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.sql.catalyst.InternalRow

import Utils.approxEqual

/**
  * 2-D Bounding Box
  *
  * @constructor Create a 2-D Bounding Box
  * @param x x-coordinate of the center
  * @param y y-coordinate of the center
  * @param width Width of the box
  * @param height Height of the box
  */
@SQLUserDefinedType(udt = classOf[Box2dType])
class Box2d(
    val x: Double,
    val y: Double,
    val width: Double,
    val height: Double
) {

  override def equals(b: Any): Boolean = {
    b match {
      case other: Box2d =>
        approxEqual(x, other.x) &&
          approxEqual(y, other.y) &&
          approxEqual(width, other.width) &&
          approxEqual(height, other.height)
      case _ => false,
    }
  }

  override def toString: String = f"Box2d(x=$x, y=$y, h=$height, w=$width)"

}

/**
  * User defined type of 2D Bouding Box
  */
private[spark] class Box2dType extends UserDefinedType[Box2d] {

  override def sqlType: DataType =
    StructType(
      Seq(
        StructField("x", DoubleType, nullable = false),
        StructField("y", DoubleType, nullable = false),
        StructField("width", DoubleType, nullable = false),
        StructField("height", DoubleType, nullable = false)
      )
    )

  override def pyUDT: String = "rikai.spark.types.geometry.Box2dType"

  override def serialize(obj: Box2d): Any = {
    val row = new GenericInternalRow(4)
    row.setDouble(0, obj.x)
    row.setDouble(1, obj.y)
    row.setDouble(2, obj.width)
    row.setDouble(3, obj.height)
    row
  }

  override def deserialize(datum: Any): Box2d = {
    datum match {
      case row: InternalRow => {
        val x = row.getDouble(0)
        val y = row.getDouble(1)
        val width = row.getDouble(2)
        val height = row.getDouble(3)
        new Box2d(x, y, width, height)
      }
    }
  }

  override def userClass: Class[Box2d] = classOf[Box2d]

  override def defaultSize: Int = 32

  override def typeName: String = "box2d"
}
