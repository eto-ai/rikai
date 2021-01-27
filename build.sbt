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
  val scalatestVersion = "3.2.0"

  Seq(
    "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
    "software.amazon.awssdk" % "s3" % awsVersion % "provided",
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

publishTo := Some(
  if (isSnapshot.value)
    Opts.resolver.sonatypeSnapshots
  else
    Opts.resolver.sonatypeStaging
)

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

