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

package ai.eto.rikai.sql.model

import ai.eto.rikai.sql.spark.SparkRunnable
import io.circe.syntax._
import io.circe.generic.encoding.DerivedAsObjectEncoder._
import org.apache.spark.sql.catalyst.FunctionIdentifier
import org.apache.spark.sql.catalyst.analysis.UnresolvedFunction
import org.apache.spark.sql.catalyst.expressions.Expression

import scala.collection.JavaConverters.mapAsJavaMap

/** A Machine Learning Model in Rikai Catalog.
  */
trait Model {

  /** Model Name */
  val name: String

  /** Model URI in the registry */
  val spec_uri: String

  /** Flavor of the model */
  val flavor: Option[String]

  /** Model Options. */
  var options: Map[String, String] = Map.empty

  /** Return options as java Map, so that it is easily accessible in Python via py4j. */
  final def javaOptions: java.util.Map[String, String] = mapAsJavaMap(options)
}

object Model {

  /** Model Name Pattern */
  val namePattern = """[a-zA-Z]\w{0,255}""".r

  @throws[ModelNameException]
  def verifyName(name: String): Unit = {
    if (!name.matches(namePattern.regex)) {
      throw new ModelNameException(s"Model name '${name}' is not valid")
    }
  }

  def serializeOptions(options: Map[String, String]): String = {
    if (options.isEmpty) ""
    else options.asJson.noSpaces
  }
}

class ModelNameException(message: String) extends Exception(message);

/** A [[Model]] that can be turned into a Spark UDF.
  *
  * @param name model name.
  * @param spec_uri the model uri.
  * @param funcName the name of a UDF which will be called when this model is invoked.
  */
class SparkUDFModel(
    val name: String,
    val spec_uri: String,
    val funcName: String,
    val flavor: Option[String],
    /** Temporary solution to address PandasUDF and UDT incompatibility */
    val preFuncName: Option[String] = None,
    val postFuncName: Option[String] = None
) extends Model
    with SparkRunnable {

  def this(name: String, spec_uri: String, funcName: String) = {
    this(name, spec_uri, funcName, None)
  }

  def this(name: String, spec_uri: String, funcName: String, flavor: String) = {
    this(name, spec_uri, funcName, Some(flavor))
  }

  override def toString: String =
    s"SparkUDFModel(name=${name}, uri=${spec_uri}, udf=${funcName})"

  /** Convert a [[Model]] to a Spark Expression in Spark SQL's logical plan. */
  override def asSpark(args: Seq[Expression]): Expression = {
    /*
     * Due to Spark Pandas UDF does not support UDT as input or output, as well as
     * supporting array of structs as output.
     *
     * Here we add a pre-processing UDF to pickle the input into binary, and
     * a post-processing UDF to unpickle the binary returned from the inference
     * function into nested structs.
     *
     * Note that these two functions (UDFs), as a hack, are not visible to the users.
     * And we will probably later to implement a new PythonRunner or other means.
     */
    val innerArgs = preFuncName match {
      case Some(name) =>
        args.map(arg =>
          UnresolvedFunction(
            new FunctionIdentifier(name),
            Seq(arg),
            isDistinct = false
          )
        )
      case None => args
    }
    val sparkFunc = UnresolvedFunction(
      new FunctionIdentifier(funcName),
      innerArgs,
      isDistinct = false
    )
    postFuncName match {
      case Some(n: String) =>
        UnresolvedFunction(
          new FunctionIdentifier(n),
          Seq(sparkFunc),
          isDistinct = false
        )
      case None => sparkFunc
    }
  }
}
