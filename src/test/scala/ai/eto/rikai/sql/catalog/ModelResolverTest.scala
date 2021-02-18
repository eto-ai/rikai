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

package ai.eto.rikai.sql.catalog

import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite

class FakeModelResolver extends ModelResolver {
  override def resolve(uri: String): Option[Model] = ???
}

class ModelResolverTest extends AnyFunSuite {

  test("Initialize TestModelResolver") {
    val spark = (SparkSession
      .builder()
      .config(
        ModelResolver.MODEL_RESOLVER_IMPL_KEY,
        "ai.eto.rikai.sql.catalog.FakeModelResolver"
      )
      .master("local[1]")
      .getOrCreate())

    val resolver = ModelResolver.get(spark)
    print(resolver)
    assert(resolver.isInstanceOf[FakeModelResolver])
  }
}
