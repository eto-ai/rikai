package ai.eto.rikai

/**
 * Rikai Spark SQL-ML extension.
 *
 * Rikai extends Spark DDL to support ML Models:
 *
 * {{{
 *   CREATE MODEL model_name
 *   [ OPTIONS (key=value, key=value, ...) ]
 *   [ AS "model_registry_uri" ]
 *
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
