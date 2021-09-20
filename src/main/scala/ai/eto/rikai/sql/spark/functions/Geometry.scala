/*
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

package ai.eto.rikai.sql.spark.functions

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.rikai.Box2d

object Geometry {

  def registerUdfs(spark: SparkSession): Unit = {

    spark.udf.register("box_area", (b: Box2d) => b.area)
  }
}
