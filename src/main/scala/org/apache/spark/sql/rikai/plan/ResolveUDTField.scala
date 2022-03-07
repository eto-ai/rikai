package org.apache.spark.sql.rikai.plan

import org.apache.spark.sql.catalyst.SQLConfHelper
import org.apache.spark.sql.catalyst.expressions.codegen.{
  CodegenContext,
  ExprCode
}
import org.apache.spark.sql.catalyst.expressions.{
  Expression,
  ExtractValue,
  NullIntolerant,
  UnaryExpression
}
import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, Project}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.types.{DataType, StructType, UserDefinedType}

case class GetUserDefinedTypeField(child: Expression, name: String)
    extends UnaryExpression
    with ExtractValue
    with NullIntolerant {

  override def dataType: DataType = ???

  override def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode = ???
}

class ResolveUDTField extends Rule[LogicalPlan] with SQLConfHelper {

  override def apply(plan: LogicalPlan): LogicalPlan = {
    println("Resolve: ResolveUDT: ", plan, plan.getClass)
    val exps = plan.expressions
    println(s"Expressions: ${exps}  resolved=${plan.resolved}")

    plan.resolveOperatorsUp {
      case p: Project => {
        println(s"This is a project: ${p} ${p.projectList(0)}")
        val first = p.projectList(0)
        println(
          s"THIS IS FIRST: ${first.dataType} ${first.dataType.getClass.getName}"
        )
        first.dataType match {
          case udt: UserDefinedType[_] => {
            println(
              s"SURE THIS IS A UDT, ${udt.sqlType.asInstanceOf[StructType]}"
            )
//            print(udt.serialize(first))
//            first.toAttribute
//            val fieldName = p.projectList(1).toString
//            val ordinal = findField(first.dataType, fieldName, conf.resolver)
//            GetStructField(first, ordinal, p.projectList(1))
            p
          }
          case _ => p
        }
      }
    }
  }
}
