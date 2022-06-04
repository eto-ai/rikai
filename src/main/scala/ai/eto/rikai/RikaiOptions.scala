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

package ai.eto.rikai

private[rikai] case class RikaiOptions(parameters: Map[String, String]) {

  /** Base path for the feature dataset
    */
  val path: String = parameters.getOrElse("path", "")

  /** Parquet block size. */
  val blockSize: Int =
    parameters
      .getOrElse("rikai.block.size", s"${RikaiOptions.defaultBlockSize}")
      .toInt

  /** Extract options */
  val options: Map[String, String] =
    parameters
      .filterKeys(k => !RikaiOptions.excludedKeys(k))
      .toMap

  /** Columns specified via df.partitionBy() */
  val partitionColumns: Option[Seq[String]] =
    parameters.get("__partition_columns") match {
      case Some(cols) =>
        Some(
          cols
            .substring(1, cols.length - 1)
            .split(",")
            .map(_.stripPrefix("\"").stripSuffix("\""))
        )
      case None => None
    }
}

private[rikai] object RikaiOptions {
  val defaultBlockSize: Int = 32 * 1024 * 1024

  /** The keys that do not write to metadata file. */
  private val excludedKeys: Set[String] = Set("path")
}
