package ai.eto.rikai.sql.spark.logical

import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan

object MlPredict extends LogicalPlan {

  override def output: Seq[Attribute] = ???

  override def children: Seq[LogicalPlan] = ???

  override def productElement(n: Int): Any = ???

  override def productArity: Int = ???

  override def canEqual(that: Any): Boolean = ???
}
