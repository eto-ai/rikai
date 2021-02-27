#  Copyright 2021 Rikai Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and

from pyspark.sql.session import SparkSession

from rikai.spark.sql import RikaiSession


def test_model_codegen_registered(spark: SparkSession):
    session = RikaiSession(spark)
    try:
        session.start()

        spark.sql(
            """CREATE MODEL foo OPTIONS (foo="str",bar=True,max_score=1.23) USING 'test://model/a/b/c'"""
        ).count()
    finally:
        session.stop()
