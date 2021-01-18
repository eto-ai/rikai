organization := "ai.eto.rikai"

name := "rikai"

version := IO.read(new File("VERSION")).trim

scalaVersion := "2.12.11"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "3.0.1",
  "software.amazon.awssdk" % "s3" % "2.14.12",
  "com.google.cloud" % "google-cloud-storage" % "1.113.0",
  "org.scalatest" %% "scalatest-funsuite" % "3.2.0" % "test"
)

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
