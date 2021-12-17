package ai.eto.rikai.sql.model.torchhub

import ai.eto.rikai.sql.model.PyImplRegistry

/** TorchHub-based Model [[Registry]].
  */
class TorchHubRegistry(val conf: Map[String, String]) extends PyImplRegistry {

  override def pyClass: String =
    "rikai.spark.sql.codegen.torchhub_registry.TorchHubRegistry"
}
