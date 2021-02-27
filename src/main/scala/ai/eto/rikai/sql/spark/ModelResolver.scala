package ai.eto.rikai.sql.spark

import ai.eto.rikai.sql.model.Model

/**
  * ModelResolver is an interface used in python
  */
trait ModelResolver {

  def resolve(uri: String): Model
}

object ModelResolverHolder {
  var resolver: ModelResolver = null

  def register(mr: ModelResolver): String = {
    println(s"XXXXXXXXX\n\n\nHey I registerd ${mr}")
    resolver = mr
    "fick"
  }
}
