package ai.eto.rikai.sql.spark.datasources

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.sql.connector.read.PartitionReader
import org.apache.spark.sql.execution.datasources.PartitionedFile
import org.apache.spark.sql.execution.datasources.v2.FilePartitionReaderFactory
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.types.StructType
import org.apache.spark.util.SerializableConfiguration
import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_videoio.VideoCapture

import org.apache.spark.sql.rikai.{ImageType, Image}

import java.net.URI

case class VideoPartitionReaderFactory(
    sqlConf: SQLConf,
    broadcastedConf: Broadcast[SerializableConfiguration],
    dataSchema: StructType,
    readDataSchema: StructType,
    partitionSchema: StructType,
    filters: Seq[Filter]
) extends FilePartitionReaderFactory {

  override def buildReader(
      file: PartitionedFile
  ): PartitionReader[InternalRow] = {
    val uri = new URI(file.filePath)
    val camera = new VideoCapture(uri.getPath)
    val frame = new Mat()
    var frame_id = 0
    var hasNext = true
    while (hasNext && (frame_id < file.start)) {
      hasNext = camera.read(frame)
      frame_id = frame_id + 1
    }

    new PartitionReader[InternalRow] {
      override def next(): Boolean = {
        frame_id = frame_id + 1
        hasNext = camera.read(frame)
        hasNext && (frame_id < file.start + file.length)
      }

      override def get(): InternalRow = {
        val row = new GenericInternalRow(2)
        row.setLong(0, frame_id)
        val image =
          new Image(data = Some(frame.data().getStringBytes), uri = None)
        val imageType = new ImageType
        row.update(1, imageType.serialize(image))
        row
      }

      override def close(): Unit = {
        camera.release()
      }
    }
  }
}
