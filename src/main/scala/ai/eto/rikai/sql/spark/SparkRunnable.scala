package ai.eto.rikai.sql.spark

import org.apache.spark.sql.catalyst.expressions.Expression

/**
  * Make [[ai.eto.rikai.sql.model.Model]] runnable on Spark.
  *
  *  For a '''ML_PREDICT''' expression in Spark SQL,
  *
  *  {{{
  *    SELECT ML_PREDICT(model_zoo, col1, col2, col3) FROM t1
  *  }}}
  *
  *  It generates a LogicalPlan equivalent to
  *
  *  {{{
  *    SELECT <Model{model_zoo}.asSpark(col1, col2, col3)> FROM t1
  *  }}}
  *
  * @example
  *
  * To implement a [[ai.eto.rikai.sql.model.Model]] for '''`RegistryFoo`''':
  *
  * {{{
  *   class FooModel(name, uri) extends Model with SparkRunnable {
  *
  *       /** Use a Spark UDF with the same name to run RegistryFoo's model */
  *      def asSpark(args: Seq[Expression]) : Expression = {
  *          UnresolvedFunction(
  *             new FunctionIdentifier(s"\${name}"),
  *             arguments,
  *             isDistinct = false,
  *             Option.empty
  *          )
  *      }
  * }}}
  */
trait SparkRunnable {

  /** Convert a [[ai.eto.rikai.sql.model.Model]] to a Spark Expression in Spark SQL's logical plan.
    */
  // Use `asSpark` instead of `expr()`/`expression()` to avoid name conflict
  // with the other SQL engine implementations.
  def asSpark(args: Seq[Expression]): Expression
}
