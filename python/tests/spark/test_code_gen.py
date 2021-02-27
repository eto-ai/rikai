from pyspark.sql.session import SparkSession
from rikai.spark.sql import RikaiSession


def test_model_resolver_register(spark: SparkSession):

    session = RikaiSession(spark)
    try:
        session.start()

        spark.sql("""CREATE MODEL foo USING 'test://modke/a/b/c'""").count()
    finally:
        session.stop()
