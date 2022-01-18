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

  private case class FuncDesc(
      func: String,
      serializer: String,
      deserializer: String
  ) {
    def funcCmd: Seq[Byte] = Base64.getDecoder.decode(func)
    def serializerFunc: Seq[Byte] = Base64.getDecoder.decode(serializer)
    def deserializerFunc: Seq[Byte] = Base64.getDecoder.decode(deserializer)
  }

  private def registerUdf(
      session: SparkSession,
      cmd: Seq[Byte],
      funcName: String,
      returnType: DataType,
      evalType: Int
  ): Unit = {
    val udf =
      UserDefinedPythonFunction(
        funcName,
        PythonFunction(
          cmd,
          new java.util.HashMap[String, String](),
          List.empty[String].asJava,
          Python.pythonExec,
          Python.pythonVer,
          Seq.empty.asJava,
          null
        ),
        returnType,
        evalType,
        udfDeterministic = true
      )
    println(s"Registered function: ${funcName} ${session}")
    session.udf.registerPython(funcName, udf)
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
      Files.write(specPath, spec.asJson.toString.getBytes)
      Python.execute(
        s"""from pyspark.serializers import CloudPickleSerializer;
           |import json
           |import base64
           |spec = json.load(open("${specPath}", "r"))
           |from rikai.spark.sql.codegen import command_from_spec
           |serialize_func, func, deserialize_func, dataType = command_from_spec("${registryClassName}", spec)
           |pickle = CloudPickleSerializer()
           |with open("${path}", "w") as fobj:
           |    json.dump({
           |        "func": base64.b64encode(pickle.dumps((func.func, func.returnType))).decode('utf-8'),
           |        "serializer": base64.b64encode(pickle.dumps((serialize_func.func, serialize_func.returnType))).decode('utf-8'),
           |        "deserializer": base64.b64encode(pickle.dumps((deserialize_func.func, deserialize_func.returnType))).decode('utf-8'),
           |    }, fobj)
           |with open("${dataTypePath}", "w") as fobj:
           |    fobj.write(dataType.json())
           |""".stripMargin,
        session
      )
      val cmdJson = Files.readAllLines(path).asScala.mkString("\n")
      val cmdMap = decode[FuncDesc](cmdJson) match {
        case Left(failure) => throw failure
        case Right(json)   => json
      }

      val dataTypeJson = Files.readAllLines(dataTypePath).asScala.mkString("\n")
      val returnType = DataType.fromJson(dataTypeJson)
      val suffix: String = Random.alphanumeric.take(6).mkString
      val udfName = s"${spec.name.getOrElse("model")}_${suffix}"
      val preUdfName = s"${udfName}_pre"
      val postUdfName = s"${udfName}_post"
      registerUdf(
        session,
        cmdMap.funcCmd,
        udfName,
        BinaryType,
        PythonEvalType.SQL_SCALAR_PANDAS_ITER_UDF
      )
      registerUdf(
        session,
        cmdMap.serializerFunc,
        preUdfName,
        BinaryType,
        PythonEvalType.SQL_BATCHED_UDF
      )
      registerUdf(
        session,
        cmdMap.deserializerFunc,
        postUdfName,
        returnType,
        PythonEvalType.SQL_BATCHED_UDF
      )

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
