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

package ai.eto.rikai.sql.catalog

import org.apache.spark.sql.catalyst.FunctionIdentifier
import org.apache.spark.sql.catalyst.analysis.UnresolvedFunction
import org.apache.spark.sql.catalyst.expressions.Expression

import java.net.URI
import java.nio.file.Paths

class Model(
    val name: String,
    val path: String = "",
    var Options: Map[String, String] = Map.empty
) {

  /**
    * Convert the Model inference code into Spark Expression.
    *
    * The concrete Spark Catalyst Express, typically a UDF, will be actually executed.
    *
    * @param arguments the remained arguments passed to the model.
    *
    * @return Spark catalyst Expression to run model inference.
    */
  def expression(arguments: Seq[Expression]): Expression = {
    // TODO: use a resolver / planner to provide plugins to offer different
    // Execution method.
    // But for now, it has the simplest form.
    UnresolvedFunction(
      new FunctionIdentifier(s"${name}"),
      arguments,
      isDistinct = false,
      Option.empty
    )
  }

  override def toString: String = s"Model(${name}, path=${path})"
}

object Model {

  private val pathPrefix = "model."

  /**
    * Create a model from a model name or model URL.
    * @param name name or model URI
    * @return A created [[Model]]
    */
  def fromName(name: String): Option[Model] = {
    if (name.startsWith(pathPrefix)) {
      val uri = new URI(name.substring(pathPrefix.length))
      val filename = Paths.get(uri.toString).getFileName.toString
      Some(new Model(filename, uri.toString))
    } else {
      None
    }
  }
}
