package ai.eto.rikai.sql.model.bootstrap

import ai.eto.rikai.sql.model.PyImplRegistry
import com.typesafe.scalalogging.LazyLogging

/** BootstrapRegistry is used when no registry URI is specified
  */
object BootstrapRegistry extends PyImplRegistry with LazyLogging {

  override def pyClass: String =
    "rikai.spark.sql.codegen.bootstrap.BootstrapRegistry"
}
