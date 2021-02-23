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

package ai.eto.rikai.sql.spark.execution

import ai.eto.rikai.sql.model.{Catalog, ModelResolveException, Registry}
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.execution.command.RunnableCommand
import org.apache.spark.sql.{Row, SparkSession}

case class CreateModelCommand(
    name: String,
    uri: Option[String],
    table: Option[TableIdentifier],
    replace: Boolean,
    options: Map[String, String]
) extends RunnableCommand {

  override def run(spark: SparkSession): Seq[Row] = {
    val catalog =
      Catalog.getOrCreate(
        spark.conf.get(
          Catalog.SQL_ML_CATALOG_IMPL_KEY,
          Catalog.SQL_ML_CATALOG_IMPL_DEFAULT
        )
      )
    val model = uri match {
      case Some(u) => Registry.resolve(u, Some(name))
      case None =>
        throw new ModelResolveException(
          "Must provide URI to CREATE MODEL (for now)"
        )
    }
    catalog.createModel(model)
    Seq.empty
  }

  override def toString(): String = s"CreateModelCommand(${name}, uri=${uri})"
}
