package ai.eto.rikai.sql.spark.datasources

import java.util
import org.apache.hadoop.fs.FileStatus
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.connector.catalog.TableCapability
import org.apache.spark.sql.connector.write.{LogicalWriteInfo, WriteBuilder}
import org.apache.spark.sql.execution.datasources.FileFormat
import org.apache.spark.sql.execution.datasources.v2.FileTable
import org.apache.spark.sql.rikai.ImageType
import org.apache.spark.sql.types._
import org.apache.spark.sql.util.CaseInsensitiveStringMap

case class VideoTable(
    name: String,
    sparkSession: SparkSession,
    options: CaseInsensitiveStringMap,
    paths: Seq[String],
    userSpecifiedSchema: Option[StructType]
) extends FileTable(sparkSession, options, paths, userSpecifiedSchema) {

  override def newScanBuilder(
      options: CaseInsensitiveStringMap
  ): VideoScanBuilder =
    VideoScanBuilder(sparkSession, fileIndex, schema, dataSchema, options)

  override def newWriteBuilder(info: LogicalWriteInfo): WriteBuilder = ???

  override def capabilities(): util.Set[TableCapability] =
    util.EnumSet.of(TableCapability.BATCH_READ)

  override def inferSchema(files: Seq[FileStatus]): Option[StructType] = Some {
    StructType(
      Seq(
        StructField("frame_id", LongType, nullable = false),
        StructField("image", new ImageType, nullable = false)
      )
    )
  }

  override def formatName: String = "Video"

  override def fallbackFileFormat: Class[_ <: FileFormat] = ???
}
