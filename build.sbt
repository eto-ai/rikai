organization := "ai.eto.rikai"

name := "rikai"

version := "0.0.1"

scalaVersion := "2.12.11"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % "3.0.1",
  "io.netty" % "netty-transport-native-epoll" % "4.1.46.Final",
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
