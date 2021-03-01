package org.apache.spark.sql.rikai

import org.apache.log4j.Logger
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{SQLUserDefinedType, UserDefinedType}
import org.reflections.Reflections

import scala.collection.JavaConverters._

object RikaiTypeRegisters {

  private val logger = Logger.getLogger("RikaiTypeRegisters")

  private var rikaiTypes: Map[String, Class[_ <: UserDefinedType[_]]] =
    Map.empty

  def loadUDTs(session: SparkSession): Unit = {
    loadUDTs("org.apache.spark.sql")
    // TODO: support load 3rd Party UDTs via SparkSession.conf
  }

  def loadUDTs(packagePrefix: String): Unit = {
    val reflections = new Reflections(packagePrefix)

    val loadedUdt =
      reflections
        .getTypesAnnotatedWith(classOf[SQLUserDefinedType])
        .asScala
        .map(c => c.getAnnotation(classOf[SQLUserDefinedType]).udt())
        .map(udt =>
          udt.getDeclaredConstructor().newInstance().simpleString -> udt
        )

    logger.info(s"All collected rikai types: ${loadedUdt}")
    rikaiTypes ++= loadedUdt
  }

  def get(name: String): Option[Class[_ <: UserDefinedType[_]]] =
    rikaiTypes.get(name)
}
