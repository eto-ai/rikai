/*
 * Copyright 2022 Rikai authors
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

import ai.eto.rikai.sql.model.{
  ModelAlreadyExistException,
  ModelResolveException,
  ModelSpec,
  Registry
}
import com.typesafe.scalalogging.LazyLogging
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.{Row, SparkSession}

import scala.collection.JavaConverters._

case class CreateModelCommand(
    name: String,
    ifNotExists: Boolean,
    flavor: Option[String],
    returns: Option[String],
    uri: Option[String],
    preprocessor: Option[String],
    postprocessor: Option[String],
    table: Option[TableIdentifier],
    replace: Boolean,
    options: Map[String, String]
) extends ModelCommand
    with LazyLogging {

  /** Java-flavor interface, being calling from Rikai python codebase via Py4j.
    */
  def this(
      name: String,
      uri: String,
      returns: String,
      flavor: String,
      preprocessor: String,
      postprocessor: String,
      replace: Boolean,
      options: java.util.Map[String, String]
  ) {
    this(
      name,
      true,
      flavor = Option(flavor),
      returns = Option(returns),
      uri = Option(uri),
      preprocessor = Option(preprocessor),
      postprocessor = Option(postprocessor),
      table = None,
      replace = replace,
      options = options.asScala.toMap
    )
  }

  @throws[ModelResolveException]
  private[spark] def asSpec: ModelSpec =
    uri match {
      case Some(u) =>
        new ModelSpec(
          name = Some(name),
          uri = Registry.normalize_uri(u).toString,
          flavor = flavor,
          schema = returns,
          preprocessor = preprocessor,
          postprocessor = postprocessor,
          options = Some(options)
        )
      case None =>
        throw new ModelResolveException(
          "Must provide URI to CREATE MODEL (for now)"
        )
    }

  @throws[ModelResolveException]
  override def run(spark: SparkSession): Seq[Row] = {
    val isModelExists = catalog(spark).modelExists(name)

    if (isModelExists && ifNotExists) {
      Seq.empty
    } else if (isModelExists && !replace) {
      throw new ModelAlreadyExistException(s"Model (${name}) already exists")
    } else {
      val model = Registry.resolve(spark, asSpec)
      model.options ++= options
      if (replace) {
        catalog(spark).dropModel(name)
      }
      catalog(spark).createModel(model)
      logger.info(s"Model ${model} created")
      Seq.empty
    }
  }

  override def toString(): String = s"CreateModelCommand(${name}, uri=${uri})"
}
