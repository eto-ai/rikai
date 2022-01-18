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

package ai.eto.rikai.sql.spark

import ai.eto.rikai.sql.model.Catalog
import ai.eto.rikai.sql.spark.expressions.Predict
import ai.eto.rikai.sql.spark.parser.{RikaiExtSqlParser, RikaiSparkSQLParser}
import org.apache.spark.sql.catalyst.FunctionIdentifier
import org.apache.spark.sql.catalyst.analysis.{
  UnresolvedAttribute,
  UnresolvedFunction
}
import org.apache.spark.sql.catalyst.expressions.{Expression, ExpressionInfo}
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.rikai.expressions.{Area, IOU}
import org.apache.spark.sql.{SparkSession, SparkSessionExtensions}

/** Rikai SparkSession extensions to enable Spark SQL ML.
  */
class RikaiSparkSessionExtensions extends (SparkSessionExtensions => Unit) {

  override def apply(extensions: SparkSessionExtensions): Unit = {

    extensions.injectParser((session, parser) => {
      new RikaiExtSqlParser(
        session,
        new RikaiSparkSQLParser(session, parser)
      )
    })

    // We just use a placeholder so that later we can compile a `ML_PREDICT` expression
    // to use Models.
    extensions.injectFunction(Predict.functionDescriptor)

    extensions.injectFunction(
      new FunctionIdentifier("area"),
      new ExpressionInfo("org.apache.spark.sql.rikai.expressions", "Area"),
      (exprs: Seq[Expression]) => Area(exprs.head)
    )

    extensions.injectFunction(
      new FunctionIdentifier("iou"),
      new ExpressionInfo("org.apache.spark.sql.rikai.expressions", "IOU"),
      (exprs: Seq[Expression]) => IOU(exprs.head, exprs(1))
    )

    extensions.injectCheckRule(session => {
      println(s"Check rule: ${session}")
      _ => Unit
    })

    class Dummy(val session: SparkSession) extends Rule[LogicalPlan] {

      val catalog = Catalog.getOrCreate(session)

      override def apply(plan: LogicalPlan): LogicalPlan = {
        plan.resolveExpressions {
          case f: UnresolvedFunction
              if f.name.funcName.toLowerCase() == "ml_predict" => {

            println(s"FOUND YOU ML_PREDICT, ${f}  ${session}")

            val arguments = f.arguments
            if (arguments.size < 2) {
              throw new UnsupportedOperationException(
                s"${f.name.funcName.toUpperCase} requires at least 2 parameters" +
                  s", got ${arguments.size}"
              )
            }

            val model_name = arguments.head
            val model = model_name match {
              case arg: UnresolvedAttribute =>
                catalog.getModel(arg.name, session)
            }
            model match {
              case Some(r: SparkRunnable) => {
                r.asSpark(arguments.drop(1))
              }
              case _ =>
                throw new UnsupportedOperationException("Unsupported model")
            }
          }
        }
      }
    }

    extensions.injectResolutionRule(session => {
      println(s"Inject Dummy: session: ${session}");
      new Dummy(session)
    })

  }
}
