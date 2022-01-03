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

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataType

import java.nio.file.Files
import scala.sys.process
import scala.sys.process.{Process, ProcessLogger}

object Python {

  /** Python executor */
  lazy val pythonExec: String =
    sys.env.getOrElse(
      "PYSPARK_PYTHON",
      sys.env.getOrElse("PYSPARK_DRIVER_PYTHON", "python3")
    )

  lazy val pythonVer: String =
    Process(
      Seq(
        pythonExec,
        "-c",
        "import sys; print('%d.%d' % sys.version_info[:2])"
      )
    ).!!.trim()

  /** Executor arbitrary python code */
  def execute(code: String): Unit =
    Process(
      Seq(Python.pythonExec, "-c", code)
    ) !!

  def execute(code: String, session: SparkSession): Unit = {
    val stderr = new StringBuilder
    val stdout = new StringBuilder
    val workerEnv = session.conf.getAll.toSeq
    val status = Process(
      Seq(Python.pythonExec, "-c", code),
      None,
      workerEnv: _*
    ) ! ProcessLogger(stdout append _, stderr append _)
    print(process.stdout.toString)
    if (status != 0) {
      throw new RuntimeException(stderr.toString)
    }
  }

  /** Load UDF from the python codebase without starting PySpark */
  def loadUdf(moduleName: String, func: String, session: SparkSession): Unit = {
    val funcPicklePath = Files.createTempFile("code", ".json")
    try {
      Python.execute(
        s"""from pyspark.serializers import CloudPickleSerializer;
           |import json
           |from ${moduleName} import ${func}
           |pickle = CloudPickleSerializer()
           |with open("${funcPicklePath}", "wb") as fobj:
           |    json.dump({
           |        "cmd": pickle.dumps((func.func, func.returnType)),
           |        "returnType": func.returnType.json(),
           |        "evalType": func.evalType,
           |    }, fobj)
           |""".stripMargin,
        session
      )
      val funcJson = Files.readString(funcPicklePath)
      println(f"funcJson: ${funcJson}")

    } finally {
      Files.delete(funcPicklePath)
    }
  }
}
