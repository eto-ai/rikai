package ai.eto.rikai.sql.spark.datasources

import ai.eto.rikai.SparkTestSession
import org.apache.spark.sql.rikai.ImageType
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.scalatest.funsuite.AnyFunSuite

class VideoDataSourceTest extends AnyFunSuite with SparkTestSession {
  test("schema of video data source") {
    val schema = spark.read.format("video")
      .load("python/tests/assets/big_buck_bunny_short.mp4")
      .schema
    assert(schema === StructType(Seq(StructField("frame_id", LongType, true),
      StructField("image", new ImageType, true))))
  }

  test("count of video data") {
    val df = spark.read.format("video")
      .load("python/tests/assets/big_buck_bunny_short.mp4")
    assert(df.count() === 300)
  }
}
