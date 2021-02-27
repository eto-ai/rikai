from pyspark.sql import SparkSession


class ModelResolver(object):
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def __repr__(self):
        return "ModelResolver"

    def resolve(self, uri: str):
        pass

    def register(self):
        jvm = self.spark.sparkContext._jvm
        print("I am calling you")
        v = jvm.ai.eto.rikai.sql.spark.ModelResolverHolder.register(self)
        print(f"You telling me {v}")

    def toString(self):
        return repr(self)

    class Java:
        implements = ["ai.eto.rikai.sql.spark.ModelResolver"]
