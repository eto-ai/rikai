package ai.eto.rikai.sql.execution

import org.apache.spark.sql.execution.command.RunnableCommand
import org.apache.spark.sql.{Row, SparkSession}
import ai.eto.rikai.sql.catalog.{MLCatalog, Model}
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference}
import org.apache.spark.sql.types.StringType

case class ShowModelsCommand() extends RunnableCommand {
    override val output: Seq[Attribute] = Seq(
        AttributeReference("Model Name", StringType, nullable = false)(),
        AttributeReference("Model Path", StringType, nullable = false)()
    )

    override def run(spark: SparkSession): Seq[Row] = {
        val catalog = MLCatalog.get(spark)
        val models = catalog.listModels()
        models.map { modelIdent: Model =>
            val name = modelIdent.name
            val path = modelIdent.path
            // ToDo(bobingm): to support option
            Row(name, path)
        }
    }
}
