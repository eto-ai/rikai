import java.io.File
import scala.reflect.io.Directory

name := "rikai-ros"

libraryDependencies ++= {
    val sparkVersion = "3.1.1"

    Seq(
        "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
    )
}