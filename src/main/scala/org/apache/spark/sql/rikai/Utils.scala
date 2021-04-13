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

private[rikai] object Utils {

  val EqualPrecesion: Double = 1e-8

  /** Approximated equal of two double values
    *
    * @param x a double number
    * @param y a double number
    * @param precision the precision that consider two doubles are correct.
    */
  def approxEqual(x: Double, y: Double, precision: Double): Boolean = {
    (x - y).abs <= precision
  }

  /** Return true if two double numbers are approximately equal */
  def approxEqual(x: Double, y: Double): Boolean =
    approxEqual(x, y, EqualPrecesion)
}
