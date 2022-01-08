#  Copyright (c) 2022 Rikai Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Wraps SQL ML functionalities in Python.
"""

from pathlib import Path
from typing import Callable, Optional, Union
import logging

from pyspark.sql import SparkSession
from pyspark.sql.types import DataType


def create_model(
    name: str,
    model_uri: Union[str, Path],
    schema: Union[str, DataType],
    preprocessor: Union[str, Callable] = None,
    postprocessor: Optional[Union[str, Callable]] = None,
    replace_if_exist: Optional[bool] = False,
):
    active_session = SparkSession.getActiveSession()
    assert (
        active_session is not None
    ), "Must run create_model with an active SparkSession"
    print(active_session)

    logging.info("Register model=%s uri=%s schema=%s", name, model_uri, schema)


def list_models():
    pass
