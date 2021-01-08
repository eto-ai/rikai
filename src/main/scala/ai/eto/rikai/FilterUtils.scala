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

package ai.eto.rikai

import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.sources.EqualTo
import org.apache.spark.sql.sources.GreaterThan
import org.apache.spark.sql.sources.GreaterThanOrEqual
import org.apache.spark.sql.sources.LessThan
import org.apache.spark.sql.sources.LessThanOrEqual

/**
  * Handle Spark Filters
  */
object FilterUtils {

  /** Apply a spark Filter to a DataFrame
    *
    */
  def apply(df: DataFrame, filter: Filter): DataFrame = {
    filter match {
      case EqualTo(attribute, value) =>
        df.filter(s"$attribute = $value")
      case GreaterThan(attribute, value) => df.filter(s"$attribute > $value")
      case GreaterThanOrEqual(attribute, value) =>
        df.filter(s"$attribute >= $value")
      case LessThan(attribute, value) => df.filter(s"$attribute < $value")
      case LessThanOrEqual(attribute, value) =>
        df.filter(s"$attribute <= $value")
      case _ => df
    }
  }

}
