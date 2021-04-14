import java.io.File
import scala.reflect.io.Directory

scalaVersion := "2.12.11"
name := "rikai"

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
  val sparkVersion = "3.1.1"
  val awsVersion = "2.15.69"
  val log4jScalaVersion = "12.0"
  val log4jVersion = "2.13.0"
  val snappyVersion = "1.1.8.4" // Support Apple Silicon
  val scalatestVersion = "3.2.0"
  val circeVersion = "0.12.3"

  Seq(
    "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
    "software.amazon.awssdk" % "s3" % awsVersion % "provided",
    "org.xerial.snappy" % "snappy-java" % snappyVersion,
    "org.apache.logging.log4j" %% "log4j-api-scala" % log4jScalaVersion,
    "org.apache.logging.log4j" % "log4j-core" % log4jVersion % Runtime,
    "org.scalatest" %% "scalatest-funsuite" % scalatestVersion % "test",
    "io.circe" %% "circe-core" % circeVersion,
    "io.circe" %% "circe-generic" % circeVersion,
    "io.circe" %% "circe-parser" % circeVersion
  )
}

scalacOptions ++= Seq(
  "-encoding",
  "utf8",
  "-Xfatal-warnings",
  "-Ywarn-dead-code",
  "-deprecation",
  "-unchecked",
  "-language:implicitConversions",
  "-language:higherKinds",
  "-language:existentials",
  "-language:postfixOps",
  "-Xlint:unused"
)

Test / parallelExecution := false

antlr4PackageName in Antlr4 := Some("ai.eto.rikai.sql.spark.parser")
antlr4GenVisitor in Antlr4 := true

enablePlugins(Antlr4Plugin)

Compile / doc / scalacOptions ++= Seq(
  "-skip-packages",
  "ai.eto.rikai.sql.spark.parser"
)

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
