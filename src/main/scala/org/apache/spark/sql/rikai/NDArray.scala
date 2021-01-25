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
  * NDArray type. Storing the persisted data from numpy ndarray.
  *
  * Note that this class does not support interoperation in scala yet.
  * It will only show a array summary in "df.show()".
  * But the data is accessible from Pyspark and Pytorch / Tensorflow readers.
  *
  * @todo consider to use arrow Array for interchangable data transfer.
  *
  * @param dtype
  */
@SQLUserDefinedType(udt = classOf[NDArrayType])
class NDArray(val dtype: Byte) {

  /**
    * It will only display a summary using df.show().
    */
  override def toString: String = s"ndarray(${portableDtype}, ...)"

  /**
    * @see Python ``rikai.convert.PortableDataType``
    */
  def portableDtype: String = {
    dtype match {
      case 1  => "uint8"
      case 2  => "uint16"
      case 3  => "uint32"
      case 4  => "uint64"
      case 5  => "int8"
      case 6  => "int16"
      case 7  => "int32"
      case 8  => "int64"
      case 9  => "float"
      case 10 => "double"
      case 11 => "boolean"
    }
  }
}

private[spark] class NDArrayType extends UserDefinedType[NDArray] {

  override def sqlType: DataType =
    StructType(
      Seq(
        StructField("type", ByteType, false),
        StructField("shape", ArrayType(IntegerType, false)),
        StructField("data", BinaryType, false)
      )
    )

  override def pyUDT: String = "rikai.spark.types.NDArrayType"

  override def serialize(obj: NDArray): Any = {
    val row = new GenericInternalRow(3)
    row.setByte(0, obj.dtype)
    row
  }

  override def deserialize(datum: Any): NDArray = {
    datum match {
      case row: InternalRow => {
        val dtype = row.getByte(0)
        new NDArray(dtype)
      }
    }
  }

  override def userClass: Class[NDArray] = classOf[NDArray]

}
