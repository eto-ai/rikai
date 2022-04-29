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

import com.thoughtworks.enableIf
import com.thoughtworks.enableIf.classpathMatches
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{
  Expression,
  ExpressionDescription,
  ImplicitCastInputTypes,
  NullIntolerant,
  UnaryExpression
}
import org.apache.spark.sql.catalyst.expressions.codegen.CodegenFallback
import org.apache.spark.sql.types.{AbstractDataType, DataType, StringType}
import org.apache.spark.sql.rikai.{ImageType, Image => ImageUDT}
import org.apache.spark.unsafe.types.UTF8String

@ExpressionDescription(
  usage = """
    _FUNC_(a) - Construct a Image UDT from a URI.
  """,
  examples = """
    Examples:
      > SELECT _FUNC_('s3://path/to/image.jpg');
        Image('s3://path/to/image.jpg')
  """
)
case class Image(child: Expression)
    extends UnaryExpression
    with CodegenFallback
    with ImplicitCastInputTypes
    with NullIntolerant {

  override def inputTypes: Seq[AbstractDataType] = Seq(StringType)

  override def dataType: DataType = new ImageType

  override def prettyName: String = "image"

  override def nullSafeEval(input: Any): Any = {
    val image = new ImageUDT(input.asInstanceOf[UTF8String].toString)
    dataType.asInstanceOf[ImageType].serialize(image)
  }

  @enableIf(classpathMatches(".*spark-catalyst_2\\.\\d+-3\\.[^01]\\..*".r))
  override def withNewChildInternal(newChild: Expression): Expression =
    copy(child = newChild)
}
