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
import re

import rikai
from rikai.__version__ import version
from rikai.conf import CONF_PARQUET_BLOCK_SIZE
from rikai.spark.sql import init


def df_to_rikai(df: "pyspark.sql.DataFrame", uri: str):
    (
        df.write.format("rikai")
        .option(CONF_PARQUET_BLOCK_SIZE, rikai.options.parquet.block.size)
        .save(uri)
    )


def get_default_jar_version(use_snapshot=True):
    """
    Make it easier to reference the jar version in notebooks and conftest.

    Parameters
    ----------
    use_snapshot: bool, default True
        If True then map `*dev0` versions to `-SNAPSHOT`
    """
    pattern = re.compile(r"([\d]+.[\d]+.[\d]+)")
    match = re.search(pattern, version)
    if not match:
        raise ValueError("Ill-formed version string {}".format(version))
    match_str = match.group(1)
    if use_snapshot and (len(match_str) < len(version)):
        return match_str + "-SNAPSHOT"
    return match_str


def init_spark_session(conf=None, app_name="rikai", rikai_version=None):
    from pyspark.sql import SparkSession

    if not rikai_version:
        rikai_version = get_default_jar_version(use_snapshot=True)
    builder = (
        SparkSession.builder.appName(app_name)
        .config(
            "spark.jars.packages", "ai.eto:rikai_2.12:{}".format(rikai_version)
        )
        .config(
            "spark.sql.extensions",
            "ai.eto.rikai.sql.spark.RikaiSparkSessionExtensions",
        )
        .config(
            "spark.driver.extraJavaOptions",
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
    )
    conf = conf or {}
    for k, v in conf.items():
        builder = builder.config(k, v)
    session = builder.master("local[2]").getOrCreate()
    init(session)
    return session
