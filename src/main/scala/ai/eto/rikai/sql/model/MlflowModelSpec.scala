package ai.eto.rikai.sql.model

class MlflowModelSpec(
                     uri:String,
                     conf:Map[String,String],
                     trackingUri:String,
                     options:Option[Map[String,Any]] = None,
                     validate: Boolean = true
                     )
