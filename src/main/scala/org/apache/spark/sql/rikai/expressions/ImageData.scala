package org.apache.spark.sql.rikai.expressions

import org.apache.spark.sql.api.java.UDF1
import org.apache.spark.sql.rikai.Image

class ImageData extends UDF1[Image, Array[Byte]] {
  override def call(image: Image): Array[Byte] = {
    image.data.orNull
  }
}
