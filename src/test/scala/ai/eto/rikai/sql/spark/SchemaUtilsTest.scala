package ai.eto.rikai.sql.spark

import org.apache.spark.sql.rikai.Box2dType
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.scalatest.funsuite.AnyFunSuite

class SchemaUtilsTest extends AnyFunSuite {

  test("parse schema") {
    val schema = StructType(
      Seq(
        StructField("id", IntegerType),
        StructField("box", new Box2dType())
      )
    )

    val schemaText = schema.simpleString
    val actual = SchemaUtils.parse(schemaText)
    assert(actual == schema)
    assert(actual.simpleString == schema.simpleString)
  }
}
