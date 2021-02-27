package ai.eto.rikai.sql.spark

import ai.eto.rikai.sql.model.Model

/**
  * [[ModelCodeGen]] to generate or import necessary code to run a model.
  */
trait ModelCodeGen {

  /** Generate code for Model */
  def generate(model: Model): Unit
}

object ModelCodeGen {
  private var codeGenerator: Option[ModelCodeGen] = None

  def register(mr: ModelCodeGen): Unit =
    codeGenerator = Some(mr)

  @throws[RuntimeException]
  def checkRegistered: Unit = {
    if (codeGenerator.isEmpty) {
      throw new RuntimeException("""ModelResolved has not been initialized.
          |Please make sure "rikai.spark.sql.RikaiSession" has started.
          |""".stripMargin)
    }
  }

  def generateCode(model: Model): Unit = {
    checkRegistered
    codeGenerator.get.generate(model)
  }
}
