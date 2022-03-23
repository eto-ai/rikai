package ai.eto.rikai.sql.model.dummy

import ai.eto.rikai.sql.model.PyImplRegistry
import com.typesafe.scalalogging.LazyLogging

/** NoURIRegistry is used when no registry URI is specified
  */
object DummyRegistry extends PyImplRegistry with LazyLogging {

  override def pyClass: String =
    "rikai.spark.sql.codegen.dummy.DummyRegistry"
}
