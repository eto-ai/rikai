package ai.eto.rikai.sql.spark.datasources

import org.apache.hadoop.fs.Path
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.connector.read.PartitionReaderFactory
import org.apache.spark.sql.execution.datasources.PartitioningAwareFileIndex
import org.apache.spark.sql.execution.datasources.v2.FileScan
import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.apache.spark.util.SerializableConfiguration

case class VideoScan(
    sparkSession: SparkSession,
    fileIndex: PartitioningAwareFileIndex,
    dataSchema: StructType,
    readDataSchema: StructType,
    readPartitionSchema: StructType,
    options: CaseInsensitiveStringMap,
    partitionFilters: Seq[Expression] = Seq.empty,
    dataFilters: Seq[Expression] = Seq.empty
) extends FileScan {

  override def isSplitable(path: Path): Boolean = true

  override def withFilters(
      partitionFilters: Seq[Expression],
      dataFilters: Seq[Expression]
  ): FileScan =
    this.copy(partitionFilters = partitionFilters, dataFilters = dataFilters)

  override def createReaderFactory(): PartitionReaderFactory = {
    val broadcastedConf = sparkSession.sparkContext.broadcast(
      new SerializableConfiguration(
        sparkSession.sparkContext.hadoopConfiguration
      )
    )

    VideoPartitionReaderFactory(
      sparkSession.sessionState.conf,
      broadcastedConf,
      dataSchema,
      readDataSchema,
      readPartitionSchema,
      Array.empty[Filter]
    )
  }
}
