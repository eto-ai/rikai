package ai.eto.rikai.sql

import org.apache.spark.internal.Logging

object ModelLoaderRegistry extends Logging {

  var loader: ModelLoader = null;

  def register(l: ModelLoader): Unit = {
    logInfo("Loading ModelLoader")
    loader = l
  }
}
