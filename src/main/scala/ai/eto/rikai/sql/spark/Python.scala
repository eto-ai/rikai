/*
 * Copyright 2021 Rikai authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.eto.rikai.sql.spark

import ai.eto.rikai.sql.model.{
  Model,
  ModelNotFoundException,
  ModelSpec,
  SparkUDFModel
}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.rikai.model.Resolver

import scala.sys.process.Process

/** [[Python]] is the callback service to call arbitrary Python code
  * in the SparkSessions' main python interpreter.
  */
trait Python {

  /** Resolve a Model from python.
    *
    * @param className the name of the python class.
    * @param spec Model spec.
    *
    * @return a Model
    */
  @throws[ModelNotFoundException]
  def resolve(
      session: SparkSession,
      className: String,
      spec: ModelSpec
  ): Model
}

object Python {
  private var python: Option[Python] = None

  /** Python executor */
  val pythonExec =
    sys.env.getOrElse(
      "PYSPARK_PYTHON",
      sys.env.getOrElse("PYSPARK_DRIVER_PYTHON", "python3")
    )

  /** Executor arbitrary python code */
  def execute(code: String): Unit =
    Process(
      Seq(Python.pythonExec, "-c", code)
    ) !!

  def register(mr: Python): Unit =
    python = Some(mr)

  @throws[RuntimeException]
  def checkRegistered: Unit = {
    if (python.isEmpty) {
      throw new RuntimeException("""ModelResolved has not been initialized.
          |Please make sure "rikai.spark.sql.init" has been called.
          |""".stripMargin)
    }
  }

  /** Resolve a Model from Python process. */
  @throws[ModelNotFoundException]
  def resolve(
      session: SparkSession,
      className: String,
      spec: ModelSpec
  ): Model = {
    print(s"The model we got now: ${spec}")

    Resolver.resolve(session, spec)

//    val path = Files.createTempFile("model-code", ".cpt")
//    try {
//      execute(s"""from pyspark.serializers import CloudPickleSerializer;
//                |from pyspark.sql.types import IntegerType
//                |from pyspark.sql.functions import udf
//                |pickle = CloudPickleSerializer()
//                |f = udf(lambda x: x + 1, IntegerType())
//                |print("FKC");
//                |with open("${path}", "wb") as fobj:
//                |    fobj.write(pickle.dumps((f.func, f.returnType)))
//                |""".stripMargin)
//      val cmd = Files.readAllBytes(path)
//      val udf = UserDefinedPythonFunction("test_sum", new PythonFunction())
//    } finally {
//      Files.delete(path)
//    }

    new SparkUDFModel("test_sum", "sum", "sumsum")
  }
}
