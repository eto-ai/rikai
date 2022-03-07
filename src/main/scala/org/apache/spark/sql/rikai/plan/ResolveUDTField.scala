package org.apache.spark.sql.rikai.plan

import org.apache.spark.sql.catalyst.expressions.codegen.{
  CodeGenerator,
  CodegenContext,
  ExprCode
}
import org.apache.spark.sql.catalyst.expressions.{
  Alias,
  Expression,
  ExtractValue,
  NullIntolerant,
  UnaryExpression
}
import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, Project}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.util.quoteIdentifier
import org.apache.spark.sql.catalyst.{InternalRow, SQLConfHelper}
import org.apache.spark.sql.types.{DataType, StructType, UserDefinedType}

case class GetUserDefinedTypeField(
    child: Expression,
    ordinal: Int,
    name: Option[String]
) extends UnaryExpression
    with ExtractValue
    with NullIntolerant {

  lazy val childSchema = child.dataType
    .asInstanceOf[UserDefinedType[_]]
    .sqlType
    .asInstanceOf[StructType]

  override def dataType: DataType = childSchema(ordinal).dataType

  def extractFieldName: String = name.getOrElse(childSchema(ordinal).name)

  override def sql: String =
    child.sql + s".${quoteIdentifier(extractFieldName)}"

  override def nullable: Boolean =
    child.nullable || childSchema(ordinal).nullable

  override def toString: String = {
    val fieldName = if (resolved) childSchema(ordinal).name else s"_$ordinal"
    s"$child.${name.getOrElse(fieldName)}"
  }

  protected override def nullSafeEval(input: Any): Any =
    input.asInstanceOf[InternalRow].get(ordinal, childSchema(ordinal).dataType)

  override def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = {
    nullSafeCodeGen(
      ctx,
      ev,
      eval => {
        if (nullable) {
          s"""
          if ($eval.isNullAt($ordinal)) {
            ${ev.isNull} = true;
          } else {
            ${ev.value} = ${CodeGenerator
            .getValue(eval, dataType, ordinal.toString)};
          }
        """
        } else {
          s"""
          ${ev.value} = ${CodeGenerator.getValue(
            eval,
            dataType,
            ordinal.toString
          )};
        """
        }
      }
    )
  }
}

class ResolveUDTField extends Rule[LogicalPlan] with SQLConfHelper {

  override def apply(plan: LogicalPlan): LogicalPlan = {
    println(s"Plan is: ${plan} ${plan.getClass}")
    plan.resolveOperatorsDown {
      case p: Project if p.projectList.length > 1 => {
        println(s"This is a project: ${p} ${p.projectList(0)}")

        val first = p.projectList(0).toAttribute
        println(
          s"THIS IS FIRST: ${first.getClass} ${first.dataType} ${first.dataType.getClass.getName}"
        )
        val field = p.projectList(1).toAttribute
        p
//        first.dataType match {
//          case udt: UserDefinedType[_] => {
//            println(
//              s"SURE THIS IS A UDT, ${udt.sqlType.asInstanceOf[StructType]}"
//            )
//            p.copy(
//              Seq(
//                Alias(
//                  GetUserDefinedTypeField(first, 1, Some(field.name)),
//                  field.name
//                )()
//              )
//            )
//          }
//          case _ => p
//        }
      }
    }
  }
}
