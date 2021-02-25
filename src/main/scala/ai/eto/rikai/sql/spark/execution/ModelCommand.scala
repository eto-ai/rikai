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

package ai.eto.rikai.sql.spark.execution

import ai.eto.rikai.sql.model.Catalog
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.execution.command.RunnableCommand
import org.apache.spark.sql.types.StringType

trait ModelCommand extends RunnableCommand {

  def catalog(session: SparkSession): Catalog =
    Catalog.getOrCreate(
      session.conf.get(
        Catalog.SQL_ML_CATALOG_IMPL_KEY,
        Catalog.SQL_ML_CATALOG_IMPL_DEFAULT
      )
    )
}

object ModelCommand {

  // Shared between "DESCRIBE MODEL" and "SHOW MODELS"
  val output = Seq(
    AttributeReference("name", StringType, nullable = false)(),
    AttributeReference("uri", StringType, nullable = false)()
  )

}
