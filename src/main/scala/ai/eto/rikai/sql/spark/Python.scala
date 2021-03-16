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

package ai.eto.rikai.sql.spark

import ai.eto.rikai.sql.model.{Model, ModelNotFoundException}

/**
  * [[Python]] is the callback service to call arbitrary Python code
  * in the SparkSessions' main python interpreter.
  */
trait Python {

  /** Generate code for a model */
  def codegen(model: Model, temporary: Boolean): Unit

  /**
    * Resolve a Model from python.
    *
    * @param uri URL for a model spec or model file.
    * @param name Optional model name. Can be empty.
    * @param options options to the model.
    *
    * @return a Model
    */
  @throws[ModelNotFoundException]
  def resolve(uri: String, name: String, options: Map[String, String]): Model
}

object Python {
  private var python: Option[Python] = None

  def register(mr: Python): Unit =
    python = Some(mr)

  @throws[RuntimeException]
  def checkRegistered: Unit = {
    if (python.isEmpty) {
      throw new RuntimeException("""ModelResolved has not been initialized.
          |Please make sure "rikai.spark.sql.init" has been called.
          |""".stripMargin)
    }
  }

  def generateCode(model: Model, temporary: Boolean = true): Unit = {
    checkRegistered
    python.get.codegen(model, temporary)
  }

  /** Resolve a Model from Python process. */
  @throws[ModelNotFoundException]
  def resolve(
      uri: String,
      name: Option[String],
      options: Map[String, String]
  ): Model = {
    checkRegistered
    python.get.resolve(uri, name.getOrElse(""), options)
  }
}
