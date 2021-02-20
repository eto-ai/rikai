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

package ai.eto.rikai.sql.model

import ai.eto.rikai.sql.spark.SparkRunnable
import org.apache.spark.sql.catalyst.FunctionIdentifier
import org.apache.spark.sql.catalyst.analysis.UnresolvedFunction
import org.apache.spark.sql.catalyst.expressions.Expression

import java.io.File
import java.net.URI

/** a FakeModel for testing */
class FakeModel(
    var name: String,
    val uri: String,
    funcName: String,
    val registry: Registry
) extends Model
    with SparkRunnable {

  def this(name: String, uri: String, registry: Registry) =
    this(name, uri, name, registry)

  /** Convert a [[ai.eto.rikai.sql.model.Model]] to a Spark Expression in Spark SQL's logical plan.
    */
  override def asSpark(args: Seq[Expression]): Expression = {
    new UnresolvedFunction(
      new FunctionIdentifier(funcName),
      arguments = args,
      isDistinct = false
    )
  }
}

/**
  * FakeRegistry for the testing purpose.
  *
  * A valid model URI is: "model://hostname/model_name"
  */
class FakeRegistry extends Registry {

  /**
    * Resolve a Model from the specific URI.
    *
    * @param uri is the model registry URI.
    *
    * @return [[Model]] if found, ``None`` otherwise.
    */
  override def resolve(uri: String): Option[Model] = {
    val parsed = URI.create(uri)
    parsed.getScheme match {
      case "model" => {
        val name = new File(
          parsed.getAuthority + "/" + parsed.getPath
        ).getName
        Some(new FakeModel(name, uri, this))
      }
      case _ => None
    }
  }
}
