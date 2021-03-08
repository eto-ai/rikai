import ReleaseTransformations._

scalaVersion := "2.12.11"

name := "rikai"

organization := "ai.eto"
homepage := Some(url("https://github.com/eto-ai/rikai"))
scmInfo := Some(
  ScmInfo(
    url("https://github.com/eto-ai/rikai"),
    "git@github.com:eto-ai/rikai.git"
  )
)
licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0"))
developers := List(
  Developer(
    "rikai",
    "developers",
    "rikai-dev@eto.ai",
    url("https://github.com/eto-ai/rikai")
  )
)
publishMavenStyle := true

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

publishTo := sonatypePublishToBundle.value

releasePublishArtifactsAction := PgpKeys.publishSigned.value
releaseProcess := Seq[ReleaseStep](
  checkSnapshotDependencies,
  inquireVersions,
  runClean,
  runTest,
  setReleaseVersion,
  commitReleaseVersion,
  tagRelease,
  releaseStepCommandAndRemaining("publishSigned"),
  releaseStepCommand("sonatypeBundleRelease"),
  setNextVersion,
  commitNextVersion,
  pushChanges
)

antlr4PackageName in Antlr4 := Some("ai.eto.rikai.sql.spark.parser")
antlr4GenVisitor in Antlr4 := true

enablePlugins(Antlr4Plugin)

Compile / doc / scalacOptions ++= Seq(
  "-skip-packages",
  "ai.eto.rikai.sql.spark.parser"
)
