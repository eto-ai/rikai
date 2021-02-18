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
#  limitations under the License.


from pyspark.sql import SparkSession


class ModelLoader:
    """Python Callback of Model Loader from Spark SQL engine.

    See Also
    --------
    ``public interface ai.eto.rikai.sql.ModelLoader`` in java
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def load(self, uri: str):
        """Dynamically load a model specified by ``uri``

        Parameters
        ----------
        uri : str
            URI for the model.
        """
        print("LEMME TRY TO LOAD MODEL")
        return path

    class Java:
        implements = ["ai.eto.rikai.sql.ModelLoader"]
