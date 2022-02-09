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

/** 3-D Bounding Box
  *
  * @constructor Create a 3-D bounding box.
  * @param center Center [[Point]] ''(x, y, z)'' of the bounding box.
  * @param width the width of the 3D box.
  * @param height the height of the 3D box.
  * @param length the length of the 3D box.
  * @param heading the heading of the bounding box (in radians).  The heading is the angle
  *        required to rotate +x to the surface normal of the box front face. It is
  *        normalized to ''[-pi, pi)''.
  *
  * @see [[https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto Waymo Dataset Spec]]
  */
@SQLUserDefinedType(udt = classOf[Box3dType])
@SerialVersionUID(1L)
class Box3d(
    val center: Point,
    val length: Double,
    val width: Double,
    val height: Double,
    val heading: Double
) extends Serializable {

  override def equals(b: Any): Boolean =
    b match {
      case other: Box3d =>
        center == other.center &&
          approxEqual(length, other.length) &&
          approxEqual(width, other.width) &&
          approxEqual(height, other.height) &&
          approxEqual(heading, other.heading)
      case _ => false
    }

  override def toString: String =
    f"Box3d(center=$center, l=$length, w=$width, h=$height, heading=$heading)"
}

/** User defined type of 3D Bounding Box
  */
class Box3dType extends UserDefinedType[Box3d] {

  override def sqlType: DataType =
    StructType(
      Seq(
        StructField(
          "center",
          PointType.sqlType,
          nullable = false
        ),
        StructField("length", DoubleType, nullable = false),
        StructField("width", DoubleType, nullable = false),
        StructField("height", DoubleType, nullable = false),
        StructField("heading", DoubleType, nullable = false)
      )
    )

  override def pyUDT: String = "rikai.spark.types.geometry.Box3dType"

  override def serialize(obj: Box3d): Any = {
    val row = new GenericInternalRow(5)
    row.update(0, PointType.serialize(obj.center))
    row.setDouble(1, obj.length)
    row.setDouble(2, obj.width)
    row.setDouble(3, obj.height)
    row.setDouble(4, obj.heading)
    row
  }

  override def deserialize(datum: Any): Box3d = {
    datum match {
      case row: InternalRow => {
        val centerRow = row.getStruct(0, 3)
        val point = PointType.deserialize(centerRow)
        val length = row.getDouble(1)
        val width = row.getDouble(2)
        val height = row.getDouble(3)
        val heading = row.getDouble(4)
        new Box3d(point, length, width, height, heading)
      }
    }
  }

  override def userClass: Class[Box3d] = classOf[Box3d]

  override def defaultSize: Int = 40

  override def typeName: String = "box3d"
}

case object Box3dType extends Box3dType
