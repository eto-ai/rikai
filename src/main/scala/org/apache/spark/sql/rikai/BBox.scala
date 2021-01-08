/*
 * Copyright 2020 Rikai authors
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
  * User defined type of 2D Bouding Box
  */
private[spark] class BBoxType extends UserDefinedType[BBox] {

  override def sqlType: DataType =
    StructType(
      Seq(
        StructField("xmin", FloatType, nullable = false),
        StructField("ymin", FloatType, nullable = false),
        StructField("xmax", FloatType, nullable = false),
        StructField("ymax", FloatType, nullable = false)
      )
    )

  override def pyUDT: String = "rikai.spark.types.BBoxType"

  override def serialize(obj: BBox): Any = {
    val row = new GenericInternalRow(4)
    row.setFloat(0, obj.xmin)
    row.setFloat(1, obj.ymin)
    row.setFloat(2, obj.xmax)
    row.setFloat(3, obj.ymax)
    row
  }

  override def deserialize(datum: Any): BBox = {
    datum match {
      case row: InternalRow => {
        val xmin = row.getFloat(0)
        val ymin = row.getFloat(1)
        val xmax = row.getFloat(2)
        val ymax = row.getFloat(3)
        new BBox(xmin, ymin, xmax, ymax)
      }
    }
  }

  override def userClass: Class[BBox] = classOf[BBox]

  override def defaultSize: Int = 32

  override def typeName: String = "bbox"
}

/**
  * 2D Bounding Box
  */
@SQLUserDefinedType(udt = classOf[BBoxType])
class BBox(
    val xmin: Float,
    val ymin: Float,
    val xmax: Float,
    val ymax: Float
) {

  override def toString: String = f"BBox($xmin, $ymin, $xmax, $ymax)"

}
