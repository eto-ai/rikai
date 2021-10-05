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

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.codegen.{
  CodegenContext,
  CodegenFallback,
  ExprCode
}
import org.apache.spark.sql.catalyst.expressions.{
  BinaryExpression,
  Expression,
  ImplicitCastInputTypes,
  NullIntolerant
}
import org.apache.spark.sql.rikai.Box2dType
import org.apache.spark.sql.types.{AbstractDataType, DataType, DoubleType}

case class Area(exprs: Seq[Expression])
    extends Expression
    with CodegenFallback {

  override def nullable: Boolean = false

  override def eval(input: InternalRow): Any = {
    val row = children.head.eval(input)
    val box = Box2dType.deserialize(row)

    box.area
  }

  override def dataType: DataType = DoubleType

  override def children: Seq[Expression] = exprs;
}

case class IOU(leftBox: Expression, rightBox: Expression)
    extends BinaryExpression
    with ImplicitCastInputTypes
    with NullIntolerant {

  override def inputTypes: Seq[AbstractDataType] = Seq(Box2dType, Box2dType)

  override def left: Expression = leftBox

  override def right: Expression = rightBox

  override def dataType: DataType = DoubleType

  override def nullSafeEval(left: Any, right: Any): Any = {
    Box2dType.deserialize(left).iou(Box2dType.deserialize(right))
  }

  override protected def doGenCode(
      ctx: CodegenContext,
      ev: ExprCode
  ): ExprCode = ???

  override def prettyName: String = "iou"
}
