package ai.eto.rikai.sql.execution

import org.apache.spark.sql.execution.command.RunnableCommand
import org.apache.spark.sql.{Row, SparkSession}
import ai.eto.rikai.sql.catalog.{MLCatalog, Model}
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeReference}
import org.apache.spark.sql.types.StringType


import org.json4s._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.write


case class ShowModelsCommand() extends RunnableCommand {
    override val output: Seq[Attribute] = Seq(
        AttributeReference("Model Name", StringType, nullable = false)(),
        AttributeReference("Model Path", StringType, nullable = false)(),
        AttributeReference("Model Options", StringType, nullable = false)()
    )

    override def run(spark: SparkSession): Seq[Row] = {
        val catalog = MLCatalog.get(spark)
        val models = catalog.listModels()
        models.map { modelIdent: Model =>
            val name = modelIdent.name
            val path = modelIdent.path
            implicit val formats = Serialization.formats(NoTypeHints)
            val option = write(modelIdent.Options)
            Row(name, path, option)
        }
    }
}
