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

import scala.collection.JavaConverters.mapAsJavaMap

/** Processor Spec.
  *
  * @param className the name of the python class/function that
  *                  process a row in ML_PREDICT.
  */
class ProcessorSpec(val className: String) {

  /** Provide access to python. */
  def getClassName: String = className
}

/** Model Spec is used to pass the create model information to
  * python ModelRegistry.
  */
class ModelSpec(
    val name: Option[String],
    val uri: String,
    val flavor: Option[String] = None,
    val schema: Option[String] = None,
    val options: Option[Map[String, String]] = None,
    val preprocessor: Option[ProcessorSpec] = None,
    val postprocessor: Option[ProcessorSpec] = None
) {

  def getName: String = name.getOrElse("")

  def getUri: String = uri

  def getSchema: String = schema.getOrElse("")

  def getOptions: java.util.Map[String, String] =
    mapAsJavaMap(options.getOrElse(Map.empty))

  /** Provide access to pre-processor via py4j. It can return Null / None in python. */
  def getPreprocessor: ProcessorSpec = preprocessor.orNull

  /** Provide access to post-processor via py4j. It can return Null / None in python. */
  def getPostprocessor: ProcessorSpec = postprocessor.orNull

  override def toString: String =
    s"ModelSpec(name=${name}, uri=${uri}, flavor=${flavor})"
}
