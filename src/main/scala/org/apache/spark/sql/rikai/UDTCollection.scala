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

package org.apache.spark.sql.rikai

import org.apache.log4j.Logger
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{SQLUserDefinedType, UserDefinedType}
import org.reflections.Reflections

import scala.collection.JavaConverters.iterableAsScalaIterableConverter

/**
  * Collections of User-defined types.
  *
  * @note
  *
  * This class is in ''o.a.spark'' package because UserDefinedType is
  * private in Spark package.
  */
object UDTCollection {

  private val logger = Logger.getLogger(UDTCollection.getClass)

  private var udtMap: Map[String, Class[_ <: UserDefinedType[_]]] =
    Map.empty

  /**
    * Load UDTs to work with Rikai schema parser.
    * .
    * @param session SparkSession
    */
  def loadUDTs(session: SparkSession): Unit = {
    // Load Rikai types
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

    logger.info(s"All collected UDTs: ${loadedUdt}")
    udtMap ++= loadedUdt
  }

  /**
    * Get the UDT class for the name.
    *
    * @param name the UDT type name.
    *
    * @return UDT class if found. Returns None otherwise.
    */
  def get(name: String): Option[Class[_ <: UserDefinedType[_]]] =
    udtMap.get(name)
}
