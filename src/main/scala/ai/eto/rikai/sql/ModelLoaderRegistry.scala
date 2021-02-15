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

import org.apache.spark.internal.Logging

/**
  * [[ModelLoader]] registry.
  */
object ModelLoaderRegistry extends Logging {

  private var loader: ModelLoader = null;

  def register(l: ModelLoader): Unit = {
    assert(loader == null)
    assert(l != null, "Registering null ModelLoader")
    logInfo("Loading ModelLoader")
    l.load("abc")
    loader = l
  }

  def get = loader
}
