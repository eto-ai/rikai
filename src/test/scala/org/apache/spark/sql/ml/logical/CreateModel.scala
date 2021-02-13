package org.apache.spark.sql.ml.logical

import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.catalyst.plans.logical.{LogicalPlan, UnaryNode}

case class CreateModel(child: LogicalPlan) extends UnaryNode {

  override def output: Seq[Attribute] = Seq.empty
}
