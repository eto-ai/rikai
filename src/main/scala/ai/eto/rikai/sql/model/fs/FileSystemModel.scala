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

package ai.eto.rikai.sql.model.fs

import ai.eto.rikai.sql.model.{Model, Registry}
import ai.eto.rikai.sql.spark.SparkRunnable
import org.apache.spark.sql.catalyst.expressions.Expression

/**
  * FileSystem-based [[Model]].
  *
  * It covers the model stored on a file system or cloud storage.
  *
  * @param name model name.
  * @param uri model URI. It can be a `.yml` file, a tar ball or a directory.
  * @param registry model registry instance.
  */
class FileSystemModel(val name: String, val uri: String, val registry: Registry)
    extends Model
    with SparkRunnable {

  /** Convert a [[Model]] to a Spark Expression in Spark SQL's logical plan. */
  override def asSpark(args: Seq[Expression]): Expression = {
    null
  }
}
