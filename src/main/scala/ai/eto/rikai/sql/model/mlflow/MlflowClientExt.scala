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

package ai.eto.rikai.sql.model.mlflow

import com.google.protobuf.InvalidProtocolBufferException
import com.typesafe.scalalogging.LazyLogging
import org.mlflow.api.proto.ModelRegistry.{
  CreateRegisteredModel,
  DeleteRegisteredModel,
  RegisteredModelTag,
  SearchRegisteredModels
}
import org.mlflow.tracking.{MlflowClientException, RikaiMlflowClient}
import org.mlflow_project.google.protobuf.Message.Builder
import org.mlflow_project.google.protobuf.MessageOrBuilder
import org.mlflow_project.google.protobuf.util.JsonFormat

import scala.collection.JavaConverters._

/** Extension to MlflowClient to add necessary APIs for Rikai */
private[mlflow] class MlflowClientExt(val trackingUri: String)
    extends LazyLogging {
  val client = new RikaiMlflowClient(trackingUri)

  private[mlflow] def searchRegisteredModels()
      : SearchRegisteredModels.Response = {
    val payload = client.sendGet("registered-models/search")
    val builder = SearchRegisteredModels.Response.newBuilder()
    logger.debug(s"Search Register Model response: ${payload}")
    MlflowClientExt.merge(payload, builder)
    builder.build()
  }

  private[mlflow] def getModel(name: String) = {
    val list = client
      .getLatestVersions(
        name,
        //TODO how to configure?
        Seq("None", "Staging", "Production", "Archived").asJava
      )
      .asScala
    list.headOption
  }

  /** Create models */
  private[mlflow] def createModel(
      name: String,
      tags: Map[String, String] = Map.empty
  ): String = {
    val requestBuilder = CreateRegisteredModel.newBuilder().setName(name)
    for ((name, value) <- tags) {
      requestBuilder.addTags(
        RegisteredModelTag.newBuilder().setKey(name).setValue(value)
      )
    }
    val request: CreateRegisteredModel =
      requestBuilder.build()
    val json = MlflowClientExt.jsonify(request)
    val respJson = client.sendPost("registered-models/create", json)
    logger.info(s"Create model response: ${respJson}")
    val response = CreateRegisteredModel.Response.newBuilder()
    MlflowClientExt.merge(respJson, response)
    response.getRegisteredModel.getName
  }

  private[mlflow] def deleteModel(name: String): Unit = {
    val request: DeleteRegisteredModel =
      DeleteRegisteredModel.newBuilder().setName(name).build()
    val payload = MlflowClientExt.jsonify(request)
    client.sendDelete(s"registered-models/delete", payload)
  }
}

protected object MlflowClientExt {

  def jsonify(message: MessageOrBuilder): String = {
    try {
      JsonFormat.printer().preservingProtoFieldNames().print(message);
    } catch {
      case e: InvalidProtocolBufferException =>
        throw new MlflowClientException(
          "Failed to serialize message " + message,
          e
        );
    }
  }

  /** Merge json payload to the protobuf builder. */
  def merge(
      jsonPayload: String,
      builder: Builder
  ) = {
    try {
      JsonFormat.parser.ignoringUnknownFields.merge(jsonPayload, builder)
    } catch {
      case e: InvalidProtocolBufferException =>
        throw new MlflowClientException(
          "Failed to serialize json " + jsonPayload + " into " + builder,
          e
        )
    }
  }
}
