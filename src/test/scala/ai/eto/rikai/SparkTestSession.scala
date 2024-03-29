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

package ai.eto.rikai

import ai.eto.rikai.sql.model.{Catalog, Registry}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfterAll, BeforeAndAfterEach, Suite}

trait SparkTestSession extends BeforeAndAfterEach with BeforeAndAfterAll {

  this: Suite =>
  lazy val spark: SparkSession = SparkSession.builder
    .config(
      "spark.sql.extensions",
      "ai.eto.rikai.sql.spark.RikaiSparkSessionExtensions"
    )
    .config(
      Catalog.SQL_ML_CATALOG_IMPL_KEY,
      Catalog.SQL_ML_CATALOG_IMPL_DEFAULT
    )
    .config(
      Registry.REGISTRY_IMPL_PREFIX + "test.impl",
      "ai.eto.rikai.sql.model.testing.TestRegistry"
    )
    .config("spark.port.maxRetries", 128)
    .master("local[*]")
    .getOrCreate

  spark.sparkContext.setLogLevel("WARN")

  override def beforeEach(): Unit = {
    Catalog.testing.clear()
  }

  def assertEqual(actual: DataFrame, expected: DataFrame): Unit = {
    assert(actual.count() == expected.count())
    assert(actual.exceptAll(expected).isEmpty)
  }

  override protected def afterAll(): Unit = {
    spark.close()
  }
}
