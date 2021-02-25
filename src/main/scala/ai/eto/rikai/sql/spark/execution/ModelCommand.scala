package ai.eto.rikai.sql.spark.execution

import ai.eto.rikai.sql.model.Catalog
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.command.RunnableCommand

trait ModelCommand extends RunnableCommand {

  def catalog(session: SparkSession): Catalog =
    Catalog.getOrCreate(
      session.conf.get(
        Catalog.SQL_ML_CATALOG_IMPL_KEY,
        Catalog.SQL_ML_CATALOG_IMPL_DEFAULT
      )
    )
}
