package ai.eto.rikai
import org.apache.hadoop.fs.Path
import org.apache.hadoop.mapreduce.Job
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.datasources.OutputWriterFactory
import org.apache.spark.sql.execution.datasources.parquet.ParquetFileFormat
import org.apache.spark.sql.types.StructType
import org.json4s._
import org.json4s.jackson.Serialization

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

  def writeMetadataFile(
      metadataFile: Path,
      sparkSession: SparkSession,
      options: RikaiOptions
  ): Unit = {
    implicit val formats: Formats = Serialization.formats(NoTypeHints)
    println("ctx ctx ctx" + sparkSession.sparkContext)
    println("hdp hdp hdp" + sparkSession.sparkContext.hadoopConfiguration)
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
    val rikaiOptions = new RikaiOptions(options.toSeq)
    setSparkOptions(sparkSession, rikaiOptions)

    /** Rikai metadata directory. */
    val rikaiDir = new Path(rikaiOptions.path, "_rikai")

    /** Metadata file name */
    val metadataFile = new Path(rikaiDir, "metadata.json")
    println("file file file" + metadataFile)
    println("sess sess sess" + sparkSession)
    println("ooooo" + rikaiOptions)
    writeMetadataFile(metadataFile, sparkSession, rikaiOptions)

    super.prepareWrite(sparkSession, job, options, dataSchema)
  }
}
