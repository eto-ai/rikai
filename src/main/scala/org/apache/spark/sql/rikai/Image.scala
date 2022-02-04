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
import org.apache.spark.unsafe.types.UTF8String

/** Image User Defined Type
  */
private[spark] class ImageType extends UserDefinedType[Image] {

  override def sqlType: DataType =
    StructType(
      Seq(
        StructField("data", BinaryType, true),
        StructField("uri", StringType, true)
      )
    )

  override def pyUDT: String = "rikai.spark.types.vision.ImageType"

  override def serialize(obj: Image): Any = {
    val row = new GenericInternalRow(2);
    obj.data.map(x => row.update(0, x))
    obj.uri.map(x => row.update(1, UTF8String.fromString(x)))
    row
  }

  override def deserialize(datum: Any): Image =
    datum match {
      case row: InternalRow =>
        new Image(
          row.isNullAt(0) match {
            case true  => None
            case false => Some(row.getBinary(0))
          },
          row.isNullAt(1) match {
            case true  => None
            case false => Some(row.getString(1).toString)
          }
        )
      case _ => throw new IllegalArgumentException()
    }

  override def userClass: Class[Image] = classOf[Image]

  override def toString(): String = "image"

  override def defaultSize: Int = 128
}

/** A image where the content are stored externally
  *
  * @param uri
  */
@SQLUserDefinedType(udt = classOf[ImageType])
@SerialVersionUID(1L)
class Image(val data: Option[Array[Byte]], val uri: Option[String])
    extends Serializable {

  def this(uri: String) = {
    this(None, Some(uri))
  }

  def this(data: Array[Byte]) {
    this(Some(data), None)
  }

  override def toString: String = s"Image('${uri.getOrElse("<embedded>")}')"
}
