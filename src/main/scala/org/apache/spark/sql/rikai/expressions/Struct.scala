/*
 * Copyright 2022 Rikai authors
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

package org.apache.spark.sql.rikai.expressions

import com.thoughtworks.enableIf
import com.thoughtworks.enableIf.classpathMatches
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.codegen.CodegenFallback
import org.apache.spark.sql.catalyst.expressions.{
  Expression,
  ExpressionDescription,
  NullIntolerant,
  UnaryExpression
}
import org.apache.spark.sql.types.{DataType, UserDefinedType}

@ExpressionDescription(
  usage = """
    _FUNC_(a) - Returns a struct from the user defined type.
  """,
  examples = """
    Examples:
      > SELECT _FUNC_(image);
       {null, "s3://foo/bar"}
  """,
  group = "struct_funcs"
)
case class ToStruct(child: Expression)
    extends UnaryExpression
    with CodegenFallback
    with NullIntolerant {

  override def prettyName: String = "to_struct"

  override def eval(input: InternalRow): Any = {
    val struct = child.eval(input)
    struct
  }

  override def dataType: DataType = child.dataType match {
    case s: UserDefinedType[_] => s.sqlType
    case _                     => child.dataType
  }

  @enableIf(classpathMatches(".*spark-catalyst_2\\.\\d+-3\\.[^01]\\..*".r))
  override def withNewChildInternal(newChild: Expression): Expression =
    copy(child = newChild)
}
