package ai.eto.rikai.sql.model.nouri

import ai.eto.rikai.sql.model.PyImplRegistry
import com.typesafe.scalalogging.LazyLogging

/** NoURIRegistry is used when no registry URI is specified
  */
object NoURIRegistry extends PyImplRegistry with LazyLogging {

  override def pyClass: String =
    "rikai.spark.sql.codegen.nouri.NoURIRegistry"
}
