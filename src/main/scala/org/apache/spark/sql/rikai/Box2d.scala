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
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.sql.rikai.Utils.approxEqual
import org.apache.spark.sql.types._

/** 2-D Bounding Box
  *
  * @constructor Create a 2-D Bounding Box.
  *
  * @param xmin x-corrdinate of the top-left of the box.
  * @param ymin y-corrdinate of the top-left of the box.
  * @param xmax x-corrdinate of the bottom-right of the box.
  * @param ymax y-corrdinate of the bottom-right of the box.
  */
@SQLUserDefinedType(udt = classOf[Box2dType])
@SerialVersionUID(1L)
class Box2d(
    val xmin: Double,
    val ymin: Double,
    val xmax: Double,
    val ymax: Double
) extends Serializable {

  override def equals(b: Any): Boolean = {
    b match {
      case other: Box2d =>
        approxEqual(xmin, other.xmin) &&
          approxEqual(ymin, other.ymin) &&
          approxEqual(xmax, other.xmax) &&
          approxEqual(ymax, other.ymax)
      case _ => false
    }
  }

  /** Return the intersection/overlap of two bounding boxes if exist.
    *
    * {{{
    *  val overlap = box1 & box2
    *  val overlap_area = overlap.map(b => b.area()).getOrElse(0);
    * }}}
    */
  def &(that: Box2d): Option[Box2d] = intersect(that)

  /** Return the intersection of two bounding boxes if exist.
    *
    * {{{
    *  val overlap = box1 intersect box2
    *  val overlap_area = overlap.map(b => b.area()).getOrElse(0);
    * }}}
    */
  def intersect(that: Box2d): Option[Box2d] = {
    if (this.xmin > that.xmin) {
      return that intersect this
    }

    if (this.xmax <= that.xmin) {
      return None
    }

    if (this.ymin > that.ymin) {
      return that intersect this
    }

    if (this.ymax <= that.ymin) {
      return None
    }

    Some(
      new Box2d(
        xmin = this.xmin max that.xmin,
        xmax = this.ymax min that.ymax,
        ymin = this.ymin max that.ymin,
        ymax = this.ymax min that.ymax
      )
    )
  }

  /** Return True of the two bounding box overlaps. */
  def overlaps(that: Box2d): Boolean = (this & that).isDefined

  /** Calculate the size of the bounding box. */
  def area: Double = (ymax - ymin) * (xmax - xmin)

  /** Calculate IOU between two Box2d bounding boxes.
    *
    * https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    */
  def iou(that: Box2d): Double = {
    val interArea: Double = (this & that).map(b => b.area).getOrElse(0);

    interArea / (this.area + that.area - interArea)
  }

  override def toString: String =
    f"Box2d(xmin=$xmin, ymin=$ymin, xmax=$xmax, ymax=$ymax)"
}

/** User defined type of 2D Bouding Box
  */
class Box2dType extends UserDefinedType[Box2d] {

  override def sqlType: DataType =
    StructType(
      Seq(
        StructField("xmin", DoubleType, nullable = false),
        StructField("ymin", DoubleType, nullable = false),
        StructField("xmax", DoubleType, nullable = false),
        StructField("ymax", DoubleType, nullable = false)
      )
    )

  override def pyUDT: String = "rikai.spark.types.geometry.Box2dType"

  override def serialize(obj: Box2d): Any = {
    val row = new GenericInternalRow(4)
    row.setDouble(0, obj.xmin)
    row.setDouble(1, obj.ymin)
    row.setDouble(2, obj.xmax)
    row.setDouble(3, obj.ymax)
    row
  }

  override def deserialize(datum: Any): Box2d = {
    datum match {
      case row: InternalRow => {
        val xmin = row.getDouble(0)
        val ymin = row.getDouble(1)
        val xmax = row.getDouble(2)
        val ymax = row.getDouble(3)
        new Box2d(xmin, ymin, xmax, ymax)
      }
    }
  }

  override def userClass: Class[Box2d] = classOf[Box2d]

  override def defaultSize: Int = 32

  override def typeName: String = "box2d"

  override def equals(other: Any): Boolean = {
    other.isInstanceOf[Box2dType]
  }
}

object Box2dType extends Box2dType
