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

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{
  GenericInternalRow,
  UnsafeArrayData
}
import org.apache.spark.sql.catalyst.util.GenericArrayData
import org.apache.spark.sql.types._

object MaskTypeEnum extends Enumeration {

  type Type = Value

  val Polygon: Type = Value(1)
  val Rle: Type = Value(2)
  val CocoRle: Type = Value(3)
}

/** Mask of an 2-D image.
  */
@SQLUserDefinedType(udt = classOf[MaskType])
class Mask(
    val maskType: MaskTypeEnum.Type,
    val polygon: Option[Array[Array[Float]]] = None,
    val height: Option[Int] = None,
    val width: Option[Int] = None,
    val rle: Option[Array[Int]] = None
) {

  override def toString: String = {
    s"Mask($maskType)"
  }
}

object Mask {

  /** Construct a Mask from polygon array */
  def fromPolygon(data: Array[Array[Float]]): Mask = {
    new Mask(MaskTypeEnum.Polygon, polygon = Some(data))
  }

  def fromRLE(data: Array[Int], height: Int, width: Int): Mask = {
    new Mask(
      MaskTypeEnum.Rle,
      height = Some(height),
      width = Some(width),
      rle = Some(data)
    )
  }

  def fromCocoRLE(data: Array[Int], height: Int, width: Int): Mask = {
    new Mask(
      MaskTypeEnum.CocoRle,
      height = Some(height),
      width = Some(width),
      rle = Some(data)
    )
  }
}

private[spark] class MaskType extends UserDefinedType[Mask] {

  override def sqlType: DataType = StructType(
    Seq(
      StructField("type", IntegerType, nullable = false),
      StructField("height", IntegerType, nullable = true),
      StructField("width", IntegerType, nullable = true),
      StructField(
        "polygon",
        ArrayType(ArrayType(FloatType))
      ),
      StructField("rle", ArrayType(IntegerType))
    )
  )

  override def pyUDT: String = "rikai.spark.types.geometry.MaskType"

  override def serialize(m: Mask): InternalRow = {
    val row = new GenericInternalRow(5)
    row.setInt(0, m.maskType.id)
    m.maskType match {
      case MaskTypeEnum.Rle | MaskTypeEnum.CocoRle =>
        row.setInt(1, m.height.get)
        row.setInt(2, m.width.get)
        row.setNullAt(3)
        row.update(4, UnsafeArrayData.fromPrimitiveArray(m.rle.get))
      case MaskTypeEnum.Polygon =>
        val arrayData = new GenericArrayData(
          m.polygon.get.map(arr => UnsafeArrayData.fromPrimitiveArray(arr))
        )
        row.update(
          3,
          arrayData
        )
        row.setNullAt(4)
      case _ => throw new NotImplementedError()
    }
    row
  }

  override def deserialize(datum: Any): Mask = {
    datum match {
      case row: InternalRow =>
        val maskType: MaskTypeEnum.Type = MaskTypeEnum(row.getInt(0))

        maskType match {
          case MaskTypeEnum.Polygon =>
            val data = row.getArray(3)
            val polygon = (0 until data.numElements())
              .map(idx => data.getArray(idx).toFloatArray())
              .toArray
            Mask.fromPolygon(polygon)
          case MaskTypeEnum.Rle =>
            val height = row.getInt(1)
            val width = row.getInt(2)
            Mask.fromRLE(
              row.getArray(4).toArray[Int](IntegerType),
              height,
              width
            )
          case MaskTypeEnum.CocoRle =>
            val height = row.getInt(1)
            val width = row.getInt(2)
            Mask.fromCocoRLE(
              row.getArray(4).toArray[Int](IntegerType),
              height,
              width
            )
        }
    }
  }

  override def userClass: Class[Mask] = classOf[Mask]

  override def toString: String = "mask"
}
