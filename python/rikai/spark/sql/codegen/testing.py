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

from typing import Dict

from pyspark.sql import SparkSession


class TestModel:
    def __init__(self, name: str, uri: str, options: Dict[str, str]):
        self.name = name
        self.uri = uri
        self.options = options

    def codegen(self, spark: SparkSession, temporary: bool):
        """Codegen for :py:class:`TestModel`

        Parameters
        ----------
        spark : SparkSession
            SparkSession

        temporary : bool
            Whether this model is generate temporary functions.
        """
        pass
