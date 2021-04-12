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

import org.apache.logging.log4j.scala.Logging
import ai.eto.rikai.sql.model.{
  Registry,
  ModelResolveException,
  ModelAlreadyExistException
}
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.{Row, SparkSession}

case class CreateModelCommand(
    name: String,
    uri: Option[String],
    table: Option[TableIdentifier],
    replace: Boolean,
    options: Map[String, String]
) extends ModelCommand
    with Logging {

  override def run(spark: SparkSession): Seq[Row] = {
    if (catalog(spark).modelExists(name)) {
      throw new ModelAlreadyExistException(s"Model (${name}) already exists")
    }
    val model = uri match {
      case Some(u) => Registry.resolve(u, Some(name), Some(options))
      case None =>
        throw new ModelResolveException(
          "Must provide URI to CREATE MODEL (for now)"
        )
    }
    model.options ++= options
    catalog(spark).createModel(model)
    logger.info(s"Model ${model} created")
    Seq.empty
  }

  override def toString(): String = s"CreateModelCommand(${name}, uri=${uri})"
}
