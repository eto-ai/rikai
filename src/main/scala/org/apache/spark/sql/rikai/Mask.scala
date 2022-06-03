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
@SerialVersionUID(1L)
class Mask(
    val maskType: MaskTypeEnum.Type,
    val width: Int,
    val height: Int,
    val polygon: Option[Array[Array[Float]]] = None,
    val rle: Option[Array[Int]] = None
) extends Serializable {

  override def toString: String = {
    s"Mask($maskType)"
  }
}

object Mask {

  /** Construct a Mask from polygon array */
  def fromPolygon(data: Array[Array[Float]], width: Int, height: Int): Mask = {
    new Mask(
      MaskTypeEnum.Polygon,
      width,
      height,
      polygon = Some(data)
    )
  }

  def fromRLE(data: Array[Int], height: Int, width: Int): Mask = {
    new Mask(
      MaskTypeEnum.Rle,
      width,
      height,
      rle = Some(data)
    )
  }

  def fromCocoRLE(data: Array[Int], height: Int, width: Int): Mask = {
    new Mask(
      MaskTypeEnum.CocoRle,
      width,
      height,
      rle = Some(data)
    )
  }
}

private[spark] class MaskType extends UserDefinedType[Mask] {

  override def sqlType: DataType = StructType(
    Seq(
      StructField("type", IntegerType, nullable = false),
      StructField("height", IntegerType, nullable = false),
      StructField("width", IntegerType, nullable = false),
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
    row.setInt(1, m.height)
    row.setInt(2, m.width)

    m.maskType match {
      case MaskTypeEnum.Rle | MaskTypeEnum.CocoRle =>
        row.setNullAt(3)
        row.update(4, UnsafeArrayData.fromPrimitiveArray(m.rle.get))
      case MaskTypeEnum.Polygon =>
        val arrayData = new GenericArrayData(
          m.polygon.get.map(arr => UnsafeArrayData.fromPrimitiveArray(arr))
        )
        row.update(3, arrayData)
        row.setNullAt(4)
      case _ => throw new NotImplementedError()
    }
    row
  }

  override def deserialize(datum: Any): Mask = {
    datum match {
      case row: InternalRow =>
        val maskType: MaskTypeEnum.Type = MaskTypeEnum(row.getInt(0))
        val height = row.getInt(1)
        val width = row.getInt(2)
        maskType match {
          case MaskTypeEnum.Polygon =>
            val data = row.getArray(3)
            val polygon = (0 until data.numElements())
              .map(idx => data.getArray(idx).toFloatArray())
              .toArray
            Mask.fromPolygon(polygon, width = width, height = height)
          case MaskTypeEnum.Rle =>
            Mask.fromRLE(
              row.getArray(4).toArray[Int](IntegerType),
              width = width,
              height = height
            )
          case MaskTypeEnum.CocoRle =>
            Mask.fromCocoRLE(
              row.getArray(4).toArray[Int](IntegerType),
              width = width,
              height = height
            )
        }
    }
  }

  override def userClass: Class[Mask] = classOf[Mask]

  override def toString: String = "mask"
}
