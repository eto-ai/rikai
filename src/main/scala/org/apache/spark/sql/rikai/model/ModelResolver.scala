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

import ai.eto.rikai.sql.model.{ModelSpec, SparkUDFModel}
import ai.eto.rikai.sql.spark.Python
import io.circe.generic.auto._
import io.circe.parser.decode
import io.circe.syntax._
import org.apache.spark.api.python.{PythonEvalType, PythonFunction}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.python.UserDefinedPythonFunction
import org.apache.spark.sql.types.{DataType, BinaryType}

import java.nio.file.Files
import java.util.Base64
import scala.collection.JavaConverters._
import scala.util.Random

object ModelResolver {

  private case class FuncDesc(func: String, ser: String, deser: String) {
    def funcCmd: Seq[Byte] = Base64.getDecoder.decode(func)
    def serCmd: Seq[Byte] = Base64.getDecoder.decode(ser)
    def deserCmd: Seq[Byte] = Base64.getDecoder.decode(deser)
  }

  def resolve(
      session: SparkSession,
      registryClassName: String,
      spec: ModelSpec
  ): SparkUDFModel = {
    val specPath = Files.createTempFile("model-spec", ".json")
    val path = Files.createTempFile("model-code", ".cpt")
    val dataTypePath = Files.createTempFile("model-type", ".json")
    try {
      Files.writeString(specPath, spec.asJson.toString)
      Python.execute(
        s"""from pyspark.serializers import CloudPickleSerializer;
                 |import json
                 |import base64
                 |spec = json.load(open("${specPath}", "r"))
                 |from rikai.spark.sql.codegen import command_from_spec
                 |ser_func, func, deser_func, dataType = command_from_spec("${registryClassName}", spec)
                 |pickle = CloudPickleSerializer()
                 |with open("${path}", "w") as fobj:
                 |    json.dump({
                 |        "func": base64.b64encode(pickle.dumps((func.func, func.returnType))).decode('utf-8'),
                 |        "ser": base64.b64encode(pickle.dumps((ser_func.func, ser_func.returnType))).decode('utf-8'),
                 |        "deser": base64.b64encode(pickle.dumps((deser_func.func, deser_func.returnType))).decode('utf-8'),
                 |    }, fobj)
                 |with open("${dataTypePath}", "w") as fobj:
                 |    fobj.write(dataType.json())
                 |""".stripMargin,
        session
      )
      val cmdJson = Files.readString(path)
      val cmdMap = decode[FuncDesc](cmdJson) match {
        case Left(failure) => throw failure
        case Right(json)   => json
      }

      val dataTypeJson = Files.readString(dataTypePath)
      val returnType = DataType.fromJson(dataTypeJson)
      val suffix = Random.alphanumeric.take(6)
      val udfName = s"${spec.name}_${suffix}"
      val udf =
        UserDefinedPythonFunction(
          udfName,
          PythonFunction(
            cmdMap.funcCmd,
            new java.util.HashMap[String, String](),
            List.empty[String].asJava,
            Python.pythonExec,
            Python.pythonVer,
            Seq.empty.asJava,
            null
          ),
          BinaryType,
          PythonEvalType.SQL_SCALAR_PANDAS_ITER_UDF,
          udfDeterministic = true
        )
      session.udf.registerPython(udfName, udf)

      val postUdfName = s"${udfName}_post"
      val postUdf =
        UserDefinedPythonFunction(
          udfName,
          PythonFunction(
            cmdMap.deserCmd,
            new java.util.HashMap[String, String](),
            List.empty[String].asJava,
            Python.pythonExec,
            Python.pythonVer,
            Seq.empty.asJava,
            null
          ),
          returnType,
          PythonEvalType.SQL_BATCHED_UDF,
          udfDeterministic = true
        )
      session.udf.registerPython(postUdfName, postUdf)

      val preUdfName = s"${udfName}_pre"
      val preUdf =
        UserDefinedPythonFunction(
          udfName,
          PythonFunction(
            cmdMap.serCmd,
            new java.util.HashMap[String, String](),
            List.empty[String].asJava,
            Python.pythonExec,
            Python.pythonVer,
            Seq.empty.asJava,
            null
          ),
          BinaryType,
          PythonEvalType.SQL_BATCHED_UDF,
          udfDeterministic = true
        )
      session.udf.registerPython(preUdfName, preUdf)

      new SparkUDFModel(
        spec.name.get,
        spec.uri,
        udfName,
        flavor = spec.flavor,
        preFuncName = Some(preUdfName),
        postFuncName = Some(postUdfName)
      )
    } finally {
      Files.delete(path)
      Files.delete(specPath)
      Files.delete(dataTypePath)
    }
  }

}
