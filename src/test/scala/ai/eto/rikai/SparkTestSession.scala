package ai.eto.rikai

import org.apache.spark.sql.SparkSession

trait SparkTestSession {

  lazy val spark = SparkSession.builder
    .config(
      "spark.sql.extensions",
      "ai.eto.rikai.sql.RikaiSparkSessionExtensions"
    )
    .master("local[*]")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")
}
