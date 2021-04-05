/*
 * Copyright 2021 Rikai authors
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

package ai.eto.rikai.sql.model.mlflow

import ai.eto.rikai.sql.model.Model
import ai.eto.rikai.sql.spark.SparkRunnable
import org.apache.spark.sql.catalyst.FunctionIdentifier
import org.apache.spark.sql.catalyst.analysis.UnresolvedFunction
import org.apache.spark.sql.catalyst.expressions.Expression

/**
  * Mlflow-based [[Model]].
  *
  * It covers the model produced by an Mlflow run
  *
  * @param name model name.
  * @param uri the mlflow run_id.
  * @param funcName the name of a UDF which will be called when this model is invoked.
  */
class MlflowModel(
    val name: String,
    val uri: String,
    val funcName: String
) extends Model
    with SparkRunnable {

  override def toString: String = s"MlflowModel(name=${name}, uri=${uri})"

  /** Convert a [[Model]] to a Spark Expression in Spark SQL's logical plan. */
  override def asSpark(args: Seq[Expression]): Expression = {
    new UnresolvedFunction(
      new FunctionIdentifier(funcName),
      arguments = args,
      isDistinct = false
    )
  }
}
