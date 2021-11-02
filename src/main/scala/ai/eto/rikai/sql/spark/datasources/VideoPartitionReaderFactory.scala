/*
 * Copyright 2021 Rikai authors
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

package ai.eto.rikai.sql.spark.datasources

import org.apache.commons.io.output.ByteArrayOutputStream
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
import org.apache.spark.sql.rikai.{Image, ImageType}
import org.bytedeco.javacv.{FFmpegFrameGrabber, Frame, Java2DFrameConverter}

import java.net.URI
import javax.imageio.ImageIO

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
    val grabber = new FFmpegFrameGrabber(uri.getPath)
    val converter = new Java2DFrameConverter
    grabber.start()
    grabber.setFrameNumber(file.start.toInt)
    var frame: Frame = null
    var frame_id = file.start

    new PartitionReader[InternalRow] {
      override def next(): Boolean = {
        frame = grabber.grabImage()
        frame_id = frame_id + 1
        frame != null && (frame_id < file.start + file.length)
      }

      override def get(): InternalRow = {
        val row = new GenericInternalRow(2)
        row.setLong(0, frame_id)
        val javaImage = converter.convert(frame)
        val bos = new ByteArrayOutputStream()
        ImageIO.write(javaImage, "png", bos)
        val image = new Image(data = Some(bos.toByteArray), uri = None)
        val imageType = new ImageType
        row.update(1, imageType.serialize(image))
        row
      }

      override def close(): Unit = {
        grabber.stop()
        converter.close()
      }
    }
  }
}
