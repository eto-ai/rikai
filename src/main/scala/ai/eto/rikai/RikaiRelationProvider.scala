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

import org.apache.spark.sql.{SQLContext, SaveMode}
import org.apache.spark.sql.sources.{
  BaseRelation,
  CreatableRelationProvider,
  DataSourceRegister,
  RelationProvider
}

/** Spark Relation Provider (source and sink) for "rikai" feature store
  */
class RikaiRelationProvider
    extends RelationProvider
    with CreatableRelationProvider
    with DataSourceRegister {

  /** {{{
    * spark.read.format("rikai").load("s3://path/to/featurestore")
    * }}}
    */
  override def shortName: String = "rikai"

  /** Rikai Relation for write
    *
    * @param sqlContext Spark SQL context
    * @param mode save mode
    * @param parameters
    * @param data Spark DataFrame to write to feature store
    * @return
    */
  override def createRelation(
      sqlContext: SQLContext,
      mode: SaveMode,
      parameters: Map[String, String],
      data: org.apache.spark.sql.DataFrame
  ): BaseRelation = {

    val conf = sqlContext.sparkSession.sparkContext.hadoopConfiguration
    // conf.set("parquet.summary.metadata.level", "ALL")
    conf.setInt("parquet.page.size.row.check.min", 3)
    conf.setInt("parquet.page.size.row.check.max", 32)

    val options = new RikaiOptions(parameters);

    val relation = new RikaiRelation(options)(sqlContext)
    relation.insert(data, true)

    relation
  }

  override def createRelation(
      sqlContext: SQLContext,
      parameters: Map[String, String]
  ): BaseRelation = {
    val options = new RikaiOptions(parameters);

    new RikaiRelation(options)(sqlContext)
  }
}
