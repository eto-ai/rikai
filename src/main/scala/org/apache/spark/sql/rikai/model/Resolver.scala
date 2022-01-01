/*
 * Copyright 2022 Rikai authors
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

package org.apache.spark.sql.rikai.model

import ai.eto.rikai.sql.model.ModelSpec
import ai.eto.rikai.sql.spark.Python
import io.circe.generic.auto._
import io.circe.syntax._
import org.apache.spark.api.python.{PythonEvalType, PythonFunction}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.python.UserDefinedPythonFunction
import org.apache.spark.sql.types.IntegerType

import java.nio.file.Files
import scala.collection.JavaConverters._

object Resolver {

  def resolve(
      session: SparkSession,
      spec: ModelSpec
  ): Unit = {
    val specClass = "rikai.spark.sql.codegen.fs.FileModelSpec"
    val specJson = spec.asJson
    print(specJson)
    val specPath = Files.createTempFile("model-spec", ".json")
    print(specPath)
    val path = Files.createTempFile("model-code", ".cpt")
    try {
      Files.writeString(specPath, spec.asJson.toString)
      Python.execute(s"""from pyspark.serializers import CloudPickleSerializer;
                 |import json
                 |spec = json.load(open("${specPath}", "r"))
                 |from rikai.spark.sql.codegen import command_from_spec
                 |func, dataType = command_from_spec("${specClass}", spec)
                 |pickle = CloudPickleSerializer()
                 |with open("${path}", "wb") as fobj:
                 |    fobj.write(pickle.dumps((func, dataType)))
                 |""".stripMargin)
      val cmd = Files.readAllBytes(path)
      val udf =
        UserDefinedPythonFunction(
          "test_sum",
          PythonFunction(
            cmd,
            new java.util.HashMap[String, String](),
            List.empty[String].asJava,
            Python.pythonExec,
            "3.9",
            Seq.empty.asJava,
            null
          ),
          IntegerType,
          PythonEvalType.SQL_SCALAR_PANDAS_ITER_UDF,
          udfDeterministic = true
        )
      session.udf.registerPython("sumsum", udf)
    } finally {
      Files.delete(path)
      Files.delete(specPath)
    }
  }

}
