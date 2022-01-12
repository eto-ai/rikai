package ai.eto.rikai

import org.apache.spark.sql.internal.SQLConf

object RikaiConf {
  val TORCHHUB_REG_ENABLED: Boolean = SQLConf.get
    .getConfString("rikai.sql.ml.registry.torchhub.enabled", "false")
    .toBoolean
}
