#  Copyright (c) 2021 Rikai Authors
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

"""
This is a temporary mechanism to wrap around pandas option machinery.
We'll need a permanent solution later on for two main reasons:
1. Users can accidentally clear options via pandas api
2. We need better type handling to help bridge jvm-python communication (GH134)
"""
import os
import tempfile

import pandas as pd

try:
    from pandas._config.config import (
        get_option,
        register_option,
        reset_option,
        set_option,
    )
except ModuleNotFoundError:
    from pandas.core.config import (
        get_option,
        register_option,
        reset_option,
        set_option,
    )

from rikai.__version__ import version as _rikai_version

options = pd.options

CONF_RIKAI_CACHEURI = "rikai.cache_uri"
DEFAULT_RIKAI_CACHEURI = os.path.join(tempfile.gettempdir(), "rikai")
os.makedirs(DEFAULT_RIKAI_CACHEURI, exist_ok=True)
register_option(CONF_RIKAI_CACHEURI, DEFAULT_RIKAI_CACHEURI)

CONF_PARQUET_BLOCK_SIZE = "parquet.block.size"
DEFAULT_ROW_GROUP_SIZE_BYTES = 32 * 1024 * 1024
register_option(CONF_PARQUET_BLOCK_SIZE, DEFAULT_ROW_GROUP_SIZE_BYTES)

CONF_RIKAI_IMAGE_DEFAULT_FORMAT = "rikai.image.default.format"
DEFAULT_IMAGE_DEFAULT_FORMAT = "PNG"
register_option(CONF_RIKAI_IMAGE_DEFAULT_FORMAT, DEFAULT_IMAGE_DEFAULT_FORMAT)

CONF_RIKAI_IO_HTTP_AGENT = "rikai.io.http_agent"
DEFAULT_RIKAI_IO_HTTP_AGENT = f"rikai/{_rikai_version}"
register_option(CONF_RIKAI_IO_HTTP_AGENT, DEFAULT_RIKAI_IO_HTTP_AGENT)

CONF_RIKAI_VIZ_COLOR = "rikai.viz.color"
DEFAULT_RIKAI_VIZ_COLOR = "red"
register_option(CONF_RIKAI_VIZ_COLOR, DEFAULT_RIKAI_VIZ_COLOR)
