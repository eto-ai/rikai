package org.apache.spark.sql.rikai.model

import ai.eto.rikai.sql.model.ModelSpec
import ai.eto.rikai.sql.spark.Python
import org.apache.spark.api.python.{PythonEvalType, PythonFunction}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.python.UserDefinedPythonFunction
import org.apache.spark.sql.types.IntegerType

import collection.JavaConverters._
import java.nio.file.Files

object Resolver {

  def resolve(
      session: SparkSession,
      spec: ModelSpec
  ): Unit = {
    val path = Files.createTempFile("model-code", ".cpt")
    try {
      Python.execute(s"""from pyspark.serializers import CloudPickleSerializer;
                 |from pyspark.sql.types import IntegerType
                 |from pyspark.sql.functions import udf
                 |pickle = CloudPickleSerializer()
                 |def pd_iter(x):
                 |    print(type(x))
                 |    return x + 1
                 |f = udf(lambda x: x + 1, IntegerType())
                 |with open("${path}", "wb") as fobj:
                 |    fobj.write(pickle.dumps((pd_iter, IntegerType())))
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
          PythonEvalType.SQL_SCALAR_PANDAS_UDF,
          udfDeterministic = true
        )
      session.udf.registerPython("sumsum", udf)
    } finally {
      Files.delete(path)
    }
  }

}
