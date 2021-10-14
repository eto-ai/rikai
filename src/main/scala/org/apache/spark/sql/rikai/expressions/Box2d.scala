/*
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

package org.apache.spark.sql.rikai.expressions

import org.apache.spark.sql.catalyst.expressions.codegen.CodegenFallback
import org.apache.spark.sql.catalyst.expressions.{
  BinaryExpression,
  Expression,
  ImplicitCastInputTypes,
  NullIntolerant,
  UnaryExpression
}
import org.apache.spark.sql.rikai.Box2dType
import org.apache.spark.sql.types.{AbstractDataType, DataType, DoubleType}

case class Area(child: Expression)
    extends UnaryExpression
    with CodegenFallback
    with ImplicitCastInputTypes
    with NullIntolerant {

  override def inputTypes: Seq[AbstractDataType] = Seq(Box2dType)

  override def nullable: Boolean = true

  override def nullSafeEval(input: Any): Any = {
    Box2dType.deserialize(input).area
  }

  override def dataType: DataType = DoubleType

  override def prettyName: String = "area"

  override def withNewChildInternal(newChild: Expression): Expression =
    copy(child = newChild)
}

case class IOU(leftBox: Expression, rightBox: Expression)
    extends BinaryExpression
    with CodegenFallback
    with ImplicitCastInputTypes
    with NullIntolerant {

  override def inputTypes: Seq[AbstractDataType] = Seq(Box2dType, Box2dType)

  override def left: Expression = leftBox

  override def right: Expression = rightBox

  override def dataType: DataType = DoubleType

  override def nullSafeEval(left: Any, right: Any): Any = {
    Box2dType.deserialize(left).iou(Box2dType.deserialize(right))
  }

  override def prettyName: String = "iou"

  override def withNewChildrenInternal(
      newLeft: Expression,
      newRight: Expression
  ): Expression =
    copy(leftBox = newLeft, rightBox = newRight)
}
