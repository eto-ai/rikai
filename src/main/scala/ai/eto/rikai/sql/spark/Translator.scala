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

package ai.eto.rikai.sql.spark

import ai.eto.rikai.sql.model.Model
import org.apache.spark.sql.catalyst.expressions.Expression

class Translator {

  /**
    * Generate Spark SQL ``Expression`` to run Model Inference.
    *
    *  For a query:
    *
    *  {{{
    *    SELECT ML_PREDICT(model_zoo, col1, col2, col3) FROM t1
    *  }}}
    *
    *  It generates a LogicalPlan that is equivalent to
    *
    *  {{{
    *    SELECT <Model(model_zoo).expr(col1, col2, col3)> FROM t1
    *  }}}
    *
    * @param model the ML model to run inference on.
    * @param arguments the list of arguments passed into the model inference code.
    * @return The generated Expression
    */
  def translate(model: Model, arguments: Seq[Expression]): Expression = {
    null
  }
}
