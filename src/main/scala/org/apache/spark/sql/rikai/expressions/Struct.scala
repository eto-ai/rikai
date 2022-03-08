package org.apache.spark.sql.rikai.expressions

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.codegen.{
  CodegenContext,
  ExprCode
}
import org.apache.spark.sql.catalyst.expressions.{
  Expression,
  ExpressionDescription,
  NullIntolerant,
  UnaryExpression
}
import org.apache.spark.sql.types.{DataType, UserDefinedType}

@ExpressionDescription(
  usage = """
    _FUNC_(a1, a2, ...) - Returns a merged array of structs that represents the cartesian product of
    all input arrays
  """,
  examples = """
    Examples:
      > SELECT _FUNC_(image);
       [{"0":1,"1":3},{"0":1,"1":4},{"0":2,"1":3},{"0":2,"1":3}]
  """,
  group = "struct_funcs"
)
case class ToStruct(child: Expression)
    extends UnaryExpression
    with NullIntolerant {

  override def prettyName: String = "to_struct"

  override def eval(input: InternalRow): Any = {
    val struct = child.eval(input)
    struct
  }

  override protected def doGenCode(
      ctx: CodegenContext,
      ev: ExprCode
  ): ExprCode = ???

  override def dataType: DataType = child.dataType match {
    case s: UserDefinedType[_] => s.sqlType
    case _                     => child.dataType
  }
}
