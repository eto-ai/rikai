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

import Utils._

/**
  * Object orientation in the 3D space
  */
@SQLUserDefinedType(udt = classOf[OrientationType])
class Orientation(
    val x: Double,
    val y: Double,
    val z: Double,
    val w: Double
) {
  override def equals(o: Any): Boolean =
    o match {
      case other: Orientation =>
        approxEqual(x, other.x) &&
          approxEqual(y, other.y) &&
          approxEqual(z, other.z) &&
          approxEqual(w, other.z)
      case _ => false
    }

  override def toString: String = s"Orientation(x=$x, y=$y, z=$z, w=$z)"
}

private[spark] class OrientationType extends UserDefinedType[Orientation] {

  override def sqlType: DataType =
    StructType(
      Seq(
        StructField("x", DoubleType, nullable = false),
        StructField("y", DoubleType, nullable = false),
        StructField("z", DoubleType, nullable = false),
        StructField("w", DoubleType, nullable = false)
      )
    )

  override def pyUDT: String = "rikai.spark.types.Orientation"

  override def serialize(obj: Orientation): Any = {
    val row = new GenericInternalRow(4)
    row.setDouble(0, obj.x)
    row.setDouble(1, obj.y)
    row.setDouble(2, obj.z)
    row.setDouble(3, obj.w)
    row
  }

  override def deserialize(datum: Any): Orientation = {
    datum match {
      case row: InternalRow => {
        val x = row.getDouble(0)
        val y = row.getDouble(1)
        val z = row.getDouble(2)
        val w = row.getDouble(3)
        new Orientation(x, y, z, w)
      }
    }
  }

  override def userClass: Class[Orientation] = classOf[Orientation]

  override def defaultSize: Int = 16

  override def typeName: String = "orientation"
}

case object OrientationType extends OrientationType
