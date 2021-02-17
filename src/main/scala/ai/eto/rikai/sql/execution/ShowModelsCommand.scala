package ai.eto.rikai.sql.execution

import org.apache.spark.sql.execution.command.RunnableCommand
import org.apache.spark.sql.{Row, SparkSession}
import ai.eto.rikai.sql.catalog.{MLCatalog, Model}

case class ShowModelsCommand() extends RunnableCommand {
    override def run(spark: SparkSession): Seq[Row] = {
        val catalog = MLCatalog.get(spark)
        val models = catalog.listModels()
        models.map { modelIdent: Model =>
            val name = modelIdent.name
            val path = modelIdent.path
            val options = modelIdent.Options.toString()
            Row(name, path, options)
        }
    }
}