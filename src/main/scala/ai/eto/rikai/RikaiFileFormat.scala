/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.eto.rikai
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapreduce.Job
import org.apache.parquet.hadoop.util.ContextUtil
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.datasources.OutputWriterFactory
import org.apache.spark.sql.execution.datasources.parquet.ParquetFileFormat
import org.apache.spark.sql.types.StructType
import org.json4s._
import org.json4s.jackson.Serialization

class RikaiFileFormat extends ParquetFileFormat {
  private def setSparkOptions(
      conf: Configuration,
      options: RikaiOptions
  ): Unit = {
    conf.setInt("parquet.page.size.row.check.min", 3)
    conf.setInt("parquet.page.size.row.check.max", 32)
    conf.setInt("parquet.block.size", options.blockSize)
  }

  def writeMetadataFile(
      metadataFile: Path,
      sparkSession: SparkSession,
      options: RikaiOptions
  ): Unit = {
    implicit val formats: Formats = Serialization.formats(NoTypeHints)
    val fs = metadataFile.getFileSystem(
      sparkSession.sparkContext.hadoopConfiguration
    )

    val outStream = fs.create(metadataFile, true)
    try {
      Serialization.write(Map("options" -> options.options), outStream)
    } finally {
      outStream.close()
    }
  }

  override def prepareWrite(
      sparkSession: SparkSession,
      job: Job,
      options: Map[String, String],
      dataSchema: StructType
  ): OutputWriterFactory = {
    val rikaiOptions = RikaiOptions(options.toSeq)
    val conf: Configuration = ContextUtil.getConfiguration(job)
    setSparkOptions(conf, rikaiOptions)

    /** Rikai metadata directory. */
    val rikaiDir = new Path(rikaiOptions.path, "_rikai")

    /** Metadata file name */
    val metadataFile = new Path(rikaiDir, "metadata.json")
    writeMetadataFile(metadataFile, sparkSession, rikaiOptions)

    super.prepareWrite(sparkSession, job, options, dataSchema)
  }
}
