/*
 * Copyright 2020 Rikai authors
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

import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Row, SQLContext, SaveMode}
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types.StructType

class RikaiRelation(val options: RikaiOptions)(
    @transient val sqlContext: SQLContext
) extends BaseRelation
    with InsertableRelation
    with PrunedFilteredScan
    with PrunedScan
    with TableScan
    with Logging {

  override def schema: StructType =
    sqlContext.read.parquet(options.path).schema

  /**
    * Full scan table
    *
    * @return
    */
  override def buildScan(): RDD[Row] = {
    sqlContext.read.parquet(options.path).rdd
  }

  /**
    * Pruned Scan
    *
    * @param requiredColumns the required columns to be loaded
    * @return
    */
  override def buildScan(requiredColumns: Array[String]): RDD[Row] =
    buildScan(requiredColumns, Array.empty[Filter])

  /**
    * Pruned and filtered scan
    *
    * @param requiredColumns
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
      .select(requiredColumns.map(col): _*)
    for (filter <- filters) {
      df = FilterUtils.apply(df, filter)
    }
    return df.rdd
  }

  /**
    * Write data
    *
    * @param data
    * @param overwrite
    */
  override def insert(
      data: org.apache.spark.sql.DataFrame,
      overwrite: Boolean
  ): Unit = {
    // println(s"Rikai writing to ${options.path}")
    var writer = data.write.format("parquet")
    if (overwrite) {
      writer = writer.mode(SaveMode.Overwrite)
    }
    val total = writer.save(options.path)

    // TODO: create schema that is usable for pytorch / tf reader
    // TODO: build index
    total
  }

}
