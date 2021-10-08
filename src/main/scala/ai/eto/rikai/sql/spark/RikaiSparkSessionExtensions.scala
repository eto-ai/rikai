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

import ai.eto.rikai.sql.spark.expressions.Predict
import ai.eto.rikai.sql.spark.parser.{RikaiExtSqlParser, RikaiSparkSQLParser}
import org.apache.spark.sql.SparkSessionExtensions
import org.apache.spark.sql.catalyst.FunctionIdentifier
import org.apache.spark.sql.catalyst.expressions.{Expression, ExpressionInfo}
import org.apache.spark.sql.rikai.expressions.{Area, CreateBox2d, IOU}

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
      (exprs: Seq[Expression]) => Area(exprs)
    )

    extensions.injectFunction(
      new FunctionIdentifier("iou"),
      new ExpressionInfo("org.apache.spark.sql.rikai.expressions", "IOU"),
      (exprs: Seq[Expression]) => IOU(exprs(0), exprs(1))
    )

    extensions.injectFunction(
      new FunctionIdentifier("box2d"),
      new ExpressionInfo(
        "org.apache.spark.sql.rikai.expressions",
        "CreateBox2d"
      ),
      (exprs: Seq[Expression]) =>
        CreateBox2d(exprs(0), exprs(1), exprs(2), exprs(3))
    )
  }
}
