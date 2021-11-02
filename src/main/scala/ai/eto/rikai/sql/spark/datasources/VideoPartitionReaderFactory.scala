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

import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.opencv_videoio.VideoCapture
import org.bytedeco.javacv.{Java2DFrameConverter, OpenCVFrameConverter}

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
    val camera = new VideoCapture(uri.getPath)
    val mat = new Mat()
    var frame_id = 0
    var hasNext = true
    while (hasNext && (frame_id < file.start)) {
      hasNext = camera.read(mat)
      frame_id = frame_id + 1
    }

    new PartitionReader[InternalRow] {
      override def next(): Boolean = {
        hasNext = camera.read(mat)
        frame_id = frame_id + 1
        hasNext && (frame_id < file.start + file.length)
      }

      override def get(): InternalRow = {
        val row = new GenericInternalRow(2)
        row.setLong(0, frame_id)
        val mat2frame = new OpenCVFrameConverter.ToMat
        val frame2java = new Java2DFrameConverter
        val javaImage = frame2java.convert(mat2frame.convert(mat))
        val bos = new ByteArrayOutputStream()
        ImageIO.write(javaImage, "png", bos)
        val image = new Image(data = Some(bos.toByteArray), uri = None)
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
