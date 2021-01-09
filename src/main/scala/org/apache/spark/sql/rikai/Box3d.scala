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

/**
  * 3-D Bounding Box
  *
  * @constructor Create a 3-D bounding box.
  * @param center Center point of the box.
  * @param width the width of the 3D box.
  * @param height the height of the 3D box.
  * @param length the length of the 3D box.
  * @param orientation the orientation of the 3D bounding box.
  *
  * @see [[https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto Waymo Dataset Spec]]
  */
class Box3d(
    val center: Point,
    val length: Double,
    val width: Double,
    val height: Double,
    val orientation: Orientation
) {
  override def toString: String =
    f"Box3d(center=$center, l=$length, w=$width, h=$height, orientation=$orientation)"
}

/**
  * User defined type of 3D Bounding Box
  */
class Box3dType extends UserDefinedType[Box3d] {

  override def sqlType: DataType =
    StructType(
      Seq(
        StructField("center", PointType, nullable = false),
        StructField("length", DoubleType, nullable = false),
        StructField("width", DoubleType, nullable = false),
        StructField("height", DoubleType, nullable = false),
        StructField("orientation", OrientationType, nullable = false)
      )
    )

  override def pyUDT: String = "rikai.spark.types.Box3dType"

  override def serialize(obj: Box3d): Any = {
    val row = new GenericInternalRow(4)
    row
  }

  override def deserialize(datum: Any): Box3d = {
    datum match {
      case row: InternalRow => {
        val centerRow = row.getStruct(0, 3)
        val point = new PointType().deserialize(centerRow)
        val length = row.getDouble(1)
        val width = row.getDouble(2)
        val height = row.getDouble(3)
        val orientationRow = row.getStruct(4, 4)
        val orientation = new OrientationType().deserialize(orientationRow)
        new Box3d(point, length, width, height, orientation)
      }
    }
  }

  override def userClass: Class[Box3d] = classOf[Box3d]

  override def defaultSize: Int = 64

  override def typeName: String = "box3d"
}

case object Box3dType extends Box3dType
