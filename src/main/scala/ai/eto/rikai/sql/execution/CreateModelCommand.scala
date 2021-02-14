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

package ai.eto.rikai.sql.execution

import ai.eto.rikai.sql.catalog.{MLCatalog, Model}
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference}
import org.apache.spark.sql.execution.command.RunnableCommand
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.{Row, SparkSession}

case class CreateModelCommand(
    name: String,
    path: Option[String],
    table: Option[TableIdentifier],
    replace: Boolean,
    options: Map[String, String]
) extends RunnableCommand {

  override val output: Seq[Attribute] = Seq(
    AttributeReference("path", StringType, nullable = true)()
  )

  override def run(spark: SparkSession): Seq[Row] = {
    val catalog = MLCatalog.get(spark)
    val model = new Model(name, path.getOrElse(""), options)
    catalog.createModel(model)
    Seq(Row(name))
  }

  override def toString(): String =
    s"CreateModel(${name}, path=${path}, table=${table}"
}
