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

/** Rikai SQL-ML extension.
  *
  * Rikai offers DDL to manipulate ML Models:
  *
  * {{{
  *   CREATE MODEL model_name
  *   [ OPTIONS (key=value, key=value, ...) ]
  *   [ AS "model_registry_uri" ]
  *
  *   # List all registered models.
  *   SHOW MODELS
  *
  *   # Describe the details of a model.
  *   (DESC | DESCRIBE) MODEL model_name
  *
  *   # Drop a Model
  *   DROP MODEL model_name
  * }}}
  *
  * A `ML_PREDICT` function is implemented to run model inference.
  *
  * {{{ SELECT id, ML_PREDICT(model_name, col1, col2, col3) as predicted FROM table }}}
  */
package object sql {}
