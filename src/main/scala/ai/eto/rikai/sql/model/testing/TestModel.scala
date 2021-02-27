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

package ai.eto.rikai.sql.model.testing

import ai.eto.rikai.sql.model.{Model, Registry}
import ai.eto.rikai.sql.spark.SparkRunnable
import org.apache.spark.sql.catalyst.FunctionIdentifier
import org.apache.spark.sql.catalyst.analysis.UnresolvedFunction
import org.apache.spark.sql.catalyst.expressions.Expression

/** a [[TestModel]] for testing */
class TestModel(
    val name: String,
    val uri: String,
    funcName: String,
    val registry: Registry
) extends Model
    with SparkRunnable {

  def this(name: String, uri: String, registry: Registry) =
    this(name, uri, name, registry)

  override def toString: String = s"TestModel(${name}, uri=${uri})"

  /**
    * Convert a [[ai.eto.rikai.sql.model.Model]] to a Spark Expression in Spark SQL's logical plan.
    */
  override def asSpark(args: Seq[Expression]): Expression = {
    new UnresolvedFunction(
      new FunctionIdentifier(funcName),
      arguments = args,
      isDistinct = false
    )
  }
}
