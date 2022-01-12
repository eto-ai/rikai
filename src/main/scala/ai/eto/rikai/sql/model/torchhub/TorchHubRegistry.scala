/*
 * Copyright (c) 2021 Rikai Authors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.eto.rikai.sql.model.torchhub

import ai.eto.rikai.RikaiConf
import ai.eto.rikai.sql.model.{
  Model,
  ModelNotFoundException,
  ModelSpec,
  PyImplRegistry
}
import org.apache.spark.sql.SparkSession

/** TorchHub-based Model [[Registry]].
  */
class TorchHubRegistry(val conf: Map[String, String]) extends PyImplRegistry {
  override def resolve(session: SparkSession, spec: ModelSpec): Model = {
    if (RikaiConf.TORCHHUB_REG_ENABLED) {
      super.resolve(session, spec)
    } else {
      throw new ModelNotFoundException(message = """
          |TorchHub Registry is disabled by default for security concerns.
          |Be cautious and set `rikai.sql.ml.registry.torchhub.enabled` to true
          |only for personal usage or testing purpose.
          |""".stripMargin)
    }
  }

  override def pyClass: String =
    "rikai.spark.sql.codegen.torchhub_registry.TorchHubRegistry"
}
