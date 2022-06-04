package ai.eto.rikai
import org.apache.hadoop.mapreduce.{Job, TaskAttemptContext}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.sql.execution.datasources.{
  OutputWriter,
  OutputWriterFactory
}
import org.apache.spark.sql.execution.datasources.parquet.ParquetFileFormat
import org.apache.spark.sql.types.StructType
import org.apache.hadoop.fs.Path
import org.json4s.jackson.Serialization
import org.json4s._

class RikaiFileFormat extends ParquetFileFormat {
  private def setSparkOptions(
      sparkSession: SparkSession,
      options: RikaiOptions
  ): Unit = {
    val conf = sparkSession.sparkContext.hadoopConfiguration
    // conf.set("parquet.summary.metadata.level", "ALL")
    conf.setInt("parquet.page.size.row.check.min", 3)
    conf.setInt("parquet.page.size.row.check.max", 32)
    conf.setInt("parquet.block.size", options.blockSize)
  }

  override def prepareWrite(
      sparkSession: SparkSession,
      job: Job,
      options: Map[String, String],
      dataSchema: StructType
  ): OutputWriterFactory = {
    val rikaiOptions = new RikaiOptions(options)
    setSparkOptions(sparkSession, rikaiOptions)
    val parquetFactory =
      super.prepareWrite(sparkSession, job, options, dataSchema)

    new OutputWriterFactory {
      override def getFileExtension(context: TaskAttemptContext): String =
        parquetFactory.getFileExtension(context)

      override def newInstance(
          path: String,
          dataSchema: StructType,
          context: TaskAttemptContext
      ): OutputWriter = new OutputWriter {
        val parquetWriter: OutputWriter =
          parquetFactory.newInstance(path, dataSchema, context)

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

        override def write(row: InternalRow): Unit = parquetWriter.write(row)

        override def close(): Unit = {

          /** Rikai metadata directory. */
          val rikaiDir = new Path(rikaiOptions.path, "_rikai")

          /** Metadata file name */
          val metadataFile = new Path(rikaiDir, "metadata.json")

          writeMetadataFile(metadataFile, sparkSession, rikaiOptions)
          parquetWriter.close()
        }
      }
    }
  }
}
