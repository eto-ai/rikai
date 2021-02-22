package ai.eto.rikai

/**
  * Rikai SQL-ML extension.
  *
  * Rikai offers DDL to manipulate ML Models:
  *
  * {{{
  *   CREATE MODEL model_name
  *   [ OPTIONS (key=value, key=value, ...) ]
  *   [ AS "model_registry_uri" ]
  *
  *   # List all registered models.
  *   SHOW MODELS
  *
  *   # Describe the details of a model.
  *   (DESC | DESCRIBE) MODEL model_name
  *
  *   # Drop a Model
  *   DROP MODEL model_name
  * }}}
  */
package object sql {}
