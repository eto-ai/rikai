package ai.eto.rikai

import org.apache.spark.sql.execution.datasources.FileFormat
import org.apache.spark.sql.execution.datasources.v2.parquet.ParquetDataSourceV2

class RikaiDataSource extends ParquetDataSourceV2 {

  override def fallbackFileFormat: Class[_ <: FileFormat] =
    classOf[RikaiFileFormat]

  override def shortName(): String = "rikai"
}
