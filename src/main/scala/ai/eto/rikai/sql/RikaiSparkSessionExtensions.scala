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

package ai.eto.rikai.sql

import ai.eto.rikai.sql.parser.RikaiExtSqlParser
import org.apache.spark.sql.SparkSessionExtensions
import org.apache.spark.sql.ml.expressions.Predict
import org.apache.spark.sql.ml.parser.RikaiSparkSQLParser

class RikaiSparkSessionExtensions extends (SparkSessionExtensions => Unit) {

  override def apply(extensions: SparkSessionExtensions): Unit = {

    extensions.injectParser((session, parser) => {
      new RikaiExtSqlParser(new RikaiSparkSQLParser(session, parser))
    })

    extensions.injectFunction(Predict.functionDescriptor)
  }
}
