import java.io.File
import scala.reflect.io.Directory

name := "rikai"
crossScalaVersions := List("2.12.16", "2.13.8")
scalaVersion := "2.12.16"
scalaBinaryVersion := scalaVersion.value.split('.').slice(0, 2).mkString(".")

val sparkVersion = settingKey[String]("Apache Spark version")
val sparkVerStr =
  settingKey[String]("Apache Spark version string like spark312")
sparkVersion := {
  sys.env.get("SPARK_VERSION") match {
    case Some(sparkVersion) => sparkVersion
    case None => "3.2.1"
  }
}
sparkVerStr := s"spark${sparkVersion.value.replace(".", "")}"

def versionFmt(out: sbtdynver.GitDescribeOutput): String = {
  val parts = out.ref.dropPrefix.split('.').toList
  val major :: minor :: patch :: rest = parts
  val nextPatchInt = patch.toInt + 1
  if (out.isSnapshot)
    s"$major.$minor.$nextPatchInt-SNAPSHOT"
  else
    out.ref.dropPrefix
}

def fallbackVersion(d: java.util.Date): String =
  s"HEAD-${sbtdynver.DynVer timestamp d}"

inThisBuild(
  List(
    organization := "ai.eto",
    homepage := Some(url("https://github.com/eto-ai/rikai")),
    licenses := List(
      "Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0")
    ),
    developers := List(
      Developer(
        "rikai-dev",
        "Rikai Developers",
        "rikai-dev@eto.ai",
        url("https://github.com/eto-ai/rikai")
      )
    ),
    version := dynverGitDescribeOutput.value
      .mkVersion(versionFmt, fallbackVersion(dynverCurrentDate.value)),
    dynver := {
      val d = new java.util.Date
      sbtdynver.DynVer
        .getGitDescribeOutput(d)
        .mkVersion(versionFmt, fallbackVersion(d))
    }
  )
)

scmInfo := Some(
  ScmInfo(
    url("https://github.com/eto-ai/rikai"),
    "git@github.com:eto-ai/rikai.git"
  )
)

libraryDependencies ++= {
  val log = sLog.value
  log.warn(s"Compiling Rikai with Spark version ${sparkVersion.value}")

  val awsVersion = "2.15.69"
  val log4jVersion = "2.17.1"
  val scalaLoggingVersion = "3.9.4"
  val snappyVersion = "1.1.8.4" // Support Apple Silicon
  val scalatestVersion = "3.2.0"
  val hadoopVersion = "3.2.3"
  val mlflowVersion = "1.21.0"
  val enableifVersion = "1.1.8"

  Seq(
    "org.scala-lang" % "scala-reflect" % scalaVersion.value % Provided,
    "com.thoughtworks.enableIf" %% "enableif" % enableifVersion % "compile-internal" exclude (
      "org.scala-lang", "scala-reflect"
    ),
    "org.apache.spark" %% "spark-sql" % sparkVersion.value % Provided,
    "org.apache.spark" %% "spark-core" % sparkVersion.value % Test classifier "tests",
    "org.apache.spark" %% "spark-hive-thriftserver" % sparkVersion.value % Test,
    "org.apache.spark" %% "spark-hive-thriftserver" % sparkVersion.value % Test classifier "tests",
    "org.apache.hadoop" % "hadoop-common" % hadoopVersion % Provided,
    "software.amazon.awssdk" % "s3" % awsVersion % Provided,
    "org.xerial.snappy" % "snappy-java" % snappyVersion,
    "org.apache.logging.log4j" % "log4j-core" % log4jVersion % Runtime,
    "com.typesafe.scala-logging" %% "scala-logging" % scalaLoggingVersion exclude (
      "org.scala-lang", "scala-reflect"
    ),
    "org.scalatest" %% "scalatest-funsuite" % scalatestVersion % Test,
    "org.mlflow" % "mlflow-client" % mlflowVersion
  )
}

libraryDependencies ++= {
  import Ordering.Implicits._
  if (VersionNumber(scalaVersion.value).numbers >= Seq(2L, 13L)) {
    Nil
  } else {
    Seq(
      compilerPlugin(
        "org.scalamacros" % "paradise" % "2.1.1" cross CrossVersion.full
      )
    )
  }
}

scalacOptions ++= Seq(
  "-encoding",
  "utf8",
  // "-Xfatal-warnings",
  "-Ywarn-dead-code",
  "-deprecation",
  "-unchecked",
  "-language:implicitConversions",
  "-language:higherKinds",
  "-language:existentials",
  "-language:postfixOps",
  "-Xlint:unused"
)

// Enable macro annotation by scalac flags for Scala 2.13
scalacOptions ++= {
  import Ordering.Implicits._
  if (VersionNumber(scalaVersion.value).numbers >= Seq(2L, 13L)) {
    Seq("-Ymacro-annotations")
  } else {
    Nil
  }
}

Test / parallelExecution := false
Test / fork := true

// https://www.scala-sbt.org/1.x/docs/Testing.html#Forking+tests
import Tests._
{
  def groupByFirst(tests: Seq[TestDefinition]) =
    tests groupBy (_.name) map {
      case (name, tests) =>
        val options = ForkOptions()
        // RegistryTest changes the mutable field of the Registry singleton
        // Let us put it in the `group_0` JVM instance
        if (name.endsWith("RegistryTest"))
          new Group("group_0", tests, SubProcess(options))
        else
          new Group("group_1", tests, SubProcess(options))
    } toSeq

  Test / testGrouping := groupByFirst( (Test / definedTests).value )
}

Antlr4 / antlr4PackageName := Some("ai.eto.rikai.sql.spark.parser")
Antlr4 / antlr4GenVisitor := true
Antlr4 / antlr4Version := {
  if ("3.1.2".equals(sparkVersion.value)) "4.8-1"
  else "4.8"
}
Antlr4 / antlr4RuntimeDependency := "org.antlr" % "antlr4-runtime" % (Antlr4 / antlr4Version).value % Provided
enablePlugins(Antlr4Plugin)

Compile / doc / scalacOptions ++= Seq(
  "-skip-packages",
  "ai.eto.rikai.sql.spark.parser"
)

assembly / assemblyJarName := s"${name.value}-${sparkVerStr.value}-assembly_${scalaBinaryVersion.value}-${version.value}.jar"
// Excluding Scala library jars, see https://github.com/sbt/sbt-assembly/tree/v1.2.0#excluding-scala-library-jars
assemblyPackageScala / assembleArtifact := false

publishLocal := {
  val ivyHome = ivyPaths.value.ivyHome.get.getCanonicalPath
  val cachePath =
    s"${ivyHome}/cache/${organization.value}/${name.value}_${scalaVersion.value.split('.').slice(0, 2).mkString(".")}"
  val cache = new Directory(new File(cachePath))
  if (cache.exists) {
    println("Rikai found in local cache. Removing...")
    assert(
      cache.deleteRecursively(),
      s"Could not delete Rikai cache ${cachePath}"
    )
    println(s"Local Rikai cache removed (${cachePath})")
  }
  publishLocal.value
}

assembly / artifact := {
  val art = (assembly / artifact).value
  art.withName(s"${name.value}-${sparkVerStr.value}-assembly")
}

addArtifact(assembly / artifact, assembly)

test in assembly := {}
