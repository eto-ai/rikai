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

package org.apache.spark.sql.hive

import ai.eto.rikai.sql.model.mlflow.SparkSessionWithMlflow
import ai.eto.rikai.sql.spark.Python
import com.typesafe.scalalogging.LazyLogging
import org.apache.hadoop.hive.conf.HiveConf.ConfVars
import org.apache.hadoop.hive.ql.metadata.Hive
import org.apache.hadoop.hive.ql.session.SessionState
import org.apache.hive.service.cli.thrift.ThriftCLIService
import org.apache.spark.sql.hive.thriftserver.{HiveThriftServer2, ServerMode}
import org.mlflow.api.proto.Service.RunInfo
import org.scalatest.concurrent.Eventually.eventually
import org.scalatest.concurrent.Waiters.{interval, timeout}
import org.scalatest.funsuite.AnyFunSuite

import java.sql.{DriverManager, Statement}
import scala.collection.JavaConverters._
import scala.concurrent.duration.DurationInt

/** Test running Rikai with ThriftServer (HiveServer2) over JDBC connection. */
class ThriftServerTest
    extends AnyFunSuite
    with SparkSessionWithMlflow
    with LazyLogging {

  /** Manage a HiveServer with Rikai SparkExtension. */
  private var hiveServer: HiveThriftServer2 = _
  private var serverPort: Int = _

  def createModels(): Unit = {
    val script = getClass.getResource("/create_models.py").getPath
    Python.run(
      Seq(
        script,
        "--mlflow-uri",
        testMlflowTrackingUri,
        "--run-id",
        run.getRunId
      )
    )
  }

  override def beforeAll() {
    super.beforeAll()

    val sqlContext = spark.sqlContext
    sqlContext.setConf(ConfVars.HIVE_SERVER2_THRIFT_PORT.varname, "0")
    sqlContext.setConf(ConfVars.HIVE_SERVER2_THRIFT_HTTP_PORT.varname, "0")
    sqlContext.setConf(
      ConfVars.HIVE_SERVER2_TRANSPORT_MODE.varname,
      ServerMode.binary.toString
    )
    sqlContext.setConf(ConfVars.HIVE_START_CLEANUP_SCRATCHDIR.varname, "true")

    hiveServer = HiveThriftServer2.startWithContext(spark.sqlContext)
    hiveServer.getServices.asScala.foreach {
      case t: ThriftCLIService =>
        serverPort = t.getPortNumber
      case _ =>
    }

    eventually(timeout(30.seconds), interval(1.seconds)) {
      withJdbcStatement { _.execute("SELECT 1") }
    }
  }

  override def afterAll(): Unit = {
    try {
      if (hiveServer != null) {
        hiveServer.stop()
      }
    } finally {
      super.afterAll()
      SessionState.detachSession()
      Hive.closeCurrent()
    }
  }

  private def jdbcUri: String = s"jdbc:hive2://localhost:$serverPort/"
  protected def user: String = System.getProperty("user.name")

  protected def withJdbcStatement(fs: (Statement => Unit)*): Unit = {
    require(
      serverPort != 0,
      "Failed to bind an actual port for HiveThriftServer2"
    )
    val connections =
      fs.map { _ => DriverManager.getConnection(jdbcUri, user, "") }
    val statements = connections.map(_.createStatement())

    try {
      statements.zip(fs).foreach { case (s, f) => f(s) }
    } finally {
      statements.foreach(_.close())
      connections.foreach(_.close())
    }
  }

  test("test ML_PREDICT on HiveServer2") {
    createModels()
    val imageUri = getClass.getResource("/000000304150.jpg").getPath

    spark.sql("SHOW MODELS").show()
    withJdbcStatement(stmt => {
      val rs = stmt.executeQuery(
        s"SELECT ML_PREDICT(ssd, '${imageUri}')"
      )
      assert(rs.next())
      assert(rs.getString(1) == "ssd")
    })
  }

}
