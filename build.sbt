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
  val sparkVersion = "3.0.1"
  val awsVersion = "2.15.69"
  val circeYamlVersion = "0.13.1"
  val snappyVersion = "1.1.8.4" // Temporarily support Apple Silicon build
  val scalatestVersion = "3.2.0"

  Seq(
    "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
    "software.amazon.awssdk" % "s3" % awsVersion % "provided",
    "org.xerial.snappy" % "snappy-java" % snappyVersion,
    "io.circe" %% "circe-yaml" % circeYamlVersion,
    "org.scalatest" %% "scalatest-funsuite" % scalatestVersion % "test"
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

antlr4PackageName in Antlr4 := Some("ai.eto.rikai.sql.parser")
antlr4GenVisitor in Antlr4 := true

enablePlugins(Antlr4Plugin)

Compile / doc / scalacOptions ++= Seq(
  "-skip-packages",
  "ai.eto.rikai.sql.parser"
)

javaOptions in Test ++= Seq(
  "-Dspark.ui.enabled=false",
  "-Dspark.ui.showConsoleProgress=false",
  "-Dspark.sql.shuffle.partitions=5",
  "-Ddelta.log.cacheSize=3",
  "-Dspark.sql.sources.parallelPartitionDiscovery.parallelism=5",
  "-Xmx1024m"
)
