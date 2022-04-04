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

import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Row, SQLContext, SaveMode}
import org.json4s.jackson.Serialization
import org.json4s._

class RikaiRelation(val options: RikaiOptions)(
    @transient val sqlContext: SQLContext
) extends BaseRelation
    with InsertableRelation
    with PrunedFilteredScan
    with PrunedScan
    with TableScan
    with Logging {

  /** Rikai metadata directory. */
  val rikaiDir = new Path(options.path, "_rikai")

  /** Metadata file name */
  val metadataFile = new Path(rikaiDir, "metadata.json")

  implicit val formats: Formats = Serialization.formats(NoTypeHints)

  override def schema: StructType =
    sqlContext.read.parquet(options.path).schema

  /** Full scan table
    *
    * @return
    */
  override def buildScan(): RDD[Row] = {
    sqlContext.read.parquet(options.path).rdd
  }

  /** Pruned Scan
    *
    * @param requiredColumns the required columns to be loaded
    * @return
    */
  override def buildScan(requiredColumns: Array[String]): RDD[Row] =
    buildScan(requiredColumns, Array.empty[Filter])

  /** Pruned and filtered scan
    *
    * @param requiredColumns the required columns to be loaded.
    * @param filters
    *
    * @return
    */
  override def buildScan(
      requiredColumns: Array[String],
      filters: Array[Filter]
  ): RDD[Row] = {
    var df = sqlContext.read
      .parquet(options.path)
      .select(requiredColumns.map(col).toSeq: _*)
    for (filter <- filters) {
      df = FilterUtils.apply(df, filter)
    }
    df.rdd
  }

  /** Write Rikai metadata to a file */
  private def writeMetadataFile(): Unit = {
    val fs = metadataFile.getFileSystem(
      sqlContext.sparkContext.hadoopConfiguration
    )

    val outStream = fs.create(metadataFile, true)
    try {
      Serialization.write(Map("a" -> 25), outStream)
    } finally {
      outStream.close()
    }
    println("OPEN METADATA: ", metadataFile, " on GC: ", fs)
  }

  /** Write data
    *
    * @param data
    * @param overwrite
    */
  override def insert(
      data: org.apache.spark.sql.DataFrame,
      overwrite: Boolean
  ): Unit = {
    var writer = data.write.format("parquet")
    if (overwrite) {
      writer = writer.mode(SaveMode.Overwrite)
    }
    val total = writer.save(options.path)

    writeMetadataFile()
    // TODO: create schema that is usable for pytorch / tf reader
    // TODO: build index
    total
  }

}
