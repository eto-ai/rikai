package ai.eto.rikai.sql.spark.datasources

import org.apache.spark.sql.connector.catalog.Table
import org.apache.spark.sql.execution.datasources.FileFormat
import org.apache.spark.sql.execution.datasources.v2.FileDataSourceV2
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

class VideoDataSourceV2 extends FileDataSourceV2 {
  override def shortName(): String = "video"

  override def toString: String = "VIDEO"

  override def getTable(options: CaseInsensitiveStringMap): Table = {
    val paths = getPaths(options)
    val tableName = getTableName(options, paths)
    val optionsWithoutPaths = getOptionsWithoutPaths(options)
    VideoTable(tableName, sparkSession, optionsWithoutPaths, paths, None)
  }

  override def getTable(
      options: CaseInsensitiveStringMap,
      schema: StructType
  ): Table = {
    val paths = getPaths(options)
    val tableName = getTableName(options, paths)
    val optionsWithoutPaths = getOptionsWithoutPaths(options)
    VideoTable(
      tableName,
      sparkSession,
      optionsWithoutPaths,
      paths,
      Some(schema)
    )
  }

  override def fallbackFileFormat: Class[_ <: FileFormat] = ???
}
