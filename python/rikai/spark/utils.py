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

import pyspark.sql.functions as F
from pyspark import keyword_only
from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param

import rikai
from rikai.__version__ import version
from rikai.conf import CONF_PARQUET_BLOCK_SIZE
from rikai.spark.functions import to_image, img_cluster


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


class Deduper(Transformer, HasInputCol, HasOutputCol):
    """
    Within Group Image Deduplication via Hierarchical
    Clustering using SSIM.

    Parameters
    ----------
    inputCol: string, default None, Required
        Name of column containing images to dedupe.
    outputCol: string, default None, Required
        Name of output column containing generated cluster ids
    groupIdCol: string, default "group_id", Optional
        Name of column containing group ids.
    threshold: float, default 0.5, Optional
        Threshold float value for clustering.

    Return
    ------
    DataFrame
        Original dataframe with additional column
        containing generated cluster ids as string values.

    Example
    -------

    >>> deduper = Deduper(inputCol="uri", outputCol="cluster_ids",
    ...                   groupIdCol="group_id", threshold=0.7)
    >>>
    >>> deduper.transform(df)
    """

    @keyword_only
    def __init__(
        self, inputCol=None, outputCol=None, groupIdCol=None, threshold=None
    ):
        super(Deduper, self).__init__()
        self.groupIdCol = Param(
            self, "groupIdCol", "Column containing group ids."
        )
        self.threshold = Param(
            self, "threshold", "Threshold float value for clustering."
        )
        self._setDefault(groupIdCol="group_id", threshold=0.5)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self, inputCol=None, outputCol=None, groupIdCol=None, threshold=None
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setGroupIdCol(self, value):
        return self._set(groupIdCol=value)

    def getGroupIdCol(self):
        return self.getOrDefault(self.groupIdCol)

    def setThreshold(self, value):
        return self._set(threshold=value)

    def getThreshold(self):
        return self.getOrDefault(self.threshold)

    def _transform(self, dataframe):
        input_col = self.getInputCol()
        out_col = self.getOutputCol()
        group_id = self.getGroupIdCol()
        threshold = self.getThreshold()

        cols = dataframe.columns

        if group_id not in cols:
            dataframe = dataframe.withColumn(group_id, F.lit("1"))

        return (
            dataframe.withColumn("image", to_image(F.col(input_col)))
            .groupBy(group_id)
            .agg(
                F.collect_list(F.col(input_col)).alias(input_col),
                F.collect_list(F.col("image")).alias("image"),
            )
            .withColumn(
                "clusters", img_cluster(F.col("image"), F.lit(threshold))
            )
            .withColumn(
                "image_cluster_id",
                F.explode(F.arrays_zip(F.col(input_col), F.col("clusters"))),
            )
            .withColumn(
                out_col,
                F.sha2(
                    F.concat(
                        F.col("image_cluster_id.clusters"), F.col(group_id)
                    ),
                    256,
                ),
            )
            .select(
                F.col("image_cluster_id.{}".format(input_col)).alias(
                    input_col
                ),
                F.col(out_col),
            )
            .join(dataframe.select(*cols), on=input_col, how="inner")
        )
