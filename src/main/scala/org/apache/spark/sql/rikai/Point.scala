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

/** A Point in 3-D space
  *
  * @constructor create a 3-D Point
  */
@SQLUserDefinedType(udt = classOf[PointType])
@SerialVersionUID(1L)
class Point(
    val x: Double,
    val y: Double,
    val z: Double
) extends Serializable {

  override def equals(p: Any): Boolean =
    p match {
      case other: Point =>
        approxEqual(x, other.x) &&
          approxEqual(y, other.y) &&
          approxEqual(z, other.z)
      case _ => false
    }

  override def toString: String = f"Point($x, $y, $z)"
}

/** User defined type for 3-D Point
  */
private[rikai] class PointType extends UserDefinedType[Point] {

  override def sqlType: DataType =
    StructType(
      Seq(
        StructField("x", DoubleType, nullable = false),
        StructField("y", DoubleType, nullable = false),
        StructField("z", DoubleType, nullable = false)
      )
    )

  override def pyUDT: String = "rikai.spark.types.geometry.PointType"

  override def serialize(obj: Point): Any = {
    val row = new GenericInternalRow(3)
    row.setDouble(0, obj.x)
    row.setDouble(1, obj.y)
    row.setDouble(2, obj.z)
    row
  }

  override def deserialize(datum: Any): Point =
    datum match {
      case row: InternalRow => {
        val x = row.getDouble(0)
        val y = row.getDouble(1)
        val z = row.getDouble(2)
        new Point(x, y, z)
      }
    }

  override def userClass: Class[Point] = classOf[Point]

  override def defaultSize: Int = 24

  override def typeName: String = "Point"
}

case object PointType extends PointType
