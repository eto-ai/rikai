/*
 * Copyright 2020 Rikai authors
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
package org.apache.spark.sql.rikai

import org.apache.spark.sql.catalyst.expressions.GenericInternalRow
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

/**
  * A VideoStream references a particular video stream at a given uri
  *
  * @param uri
  */
@SQLUserDefinedType(udt = classOf[VideoStreamType])
class VideoStream(val uri: String) {
  override def toString: String = s"VideoStream(uri='$uri')"
}

class VideoStreamType extends UserDefinedType[VideoStream] {

  override def sqlType: DataType =
    StructType(
      Seq(
        StructField("uri", StringType, false)
      )
    )

  override def pyUDT: String = "rikai.spark.types.VideoStreamType"

  override def serialize(obj: VideoStream): Any = {
    val row = new GenericInternalRow(1);
    row.update(0, UTF8String.fromString(obj.uri))
    row
  }

  override def deserialize(datum: Any): VideoStream =
    datum match {
      case row: InternalRow => new VideoStream(row.getString(0).toString())
    }

  override def userClass: Class[VideoStream] = classOf[VideoStream]

  override def toString: String = "videoStream"

  override def defaultSize: Int = 128
}

/**
  * A reference to a particular Youtube vid. A given Youtube vid is associated with many
  * content streams. These can be video streams of different formats at different bitrates,
  * audio-only streams, and video-only streams.
  *
  * @param uri
  */
@SQLUserDefinedType(udt = classOf[YouTubeVideoType])
class YouTubeVideo(val vid: String) {
  override def toString: String = s"YouTubeVideo(vid='$vid')"
}

class YouTubeVideoType extends UserDefinedType[YouTubeVideo] {

  override def sqlType: DataType =
    StructType(
      Seq(
        StructField("vid", StringType, false)
      )
    )

  override def pyUDT: String = "rikai.spark.types.YouTubeVideoType"

  override def serialize(obj: YouTubeVideo): Any = {
    val row = new GenericInternalRow(1);
    row.update(0, UTF8String.fromString(obj.vid))
    row
  }

  override def deserialize(datum: Any): YouTubeVideo =
    datum match {
      case row: InternalRow => new YouTubeVideo(row.getString(0).toString())
    }

  override def userClass: Class[YouTubeVideo] = classOf[YouTubeVideo]

  override def toString: String = "youTubeVideo"

  override def defaultSize: Int = 128
}
