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

import uuid

import pyspark.sql.types as T
import pyspark.sql.functions as F

from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer

from rikai.spark.functions.vision import tracker_match
from rikai.spark.types.geometry import Box2dType

DEFAULT_ROW_GROUP_SIZE_BYTES = 32 * 1024 * 1024


def df_to_rikai(
    df: DataFrame,
    uri: str,
    parquet_row_group_size_bytes: int = DEFAULT_ROW_GROUP_SIZE_BYTES,
):
    (
        df.write.format("rikai")
        .option("parquet.block.size", parquet_row_group_size_bytes)
        .save(uri)
    )


def match_annotations(iterator, segment_id="vid", id_col="tracker_id"):
    """
    Used by mapPartitions to iterate over the small chunks of our hierarchically-organized data.
    """

    matched_annots = []
    for idx, data in enumerate(iterator):
        data = data[1]
        if not idx:
            old_row = {idx: uuid.uuid4() for idx in range(len(data[1]))}
            old_row[segment_id] = data[0]
            pass
        annots = []
        curr_row = {segment_id: data[0]}
        if old_row[segment_id] != curr_row[segment_id]:
            old_row = {}
        if data[2] is not None:
            for ky, vl in data[2].items():
                detection = data[1][vl].asDict()
                detection[id_col] = old_row.get(ky, uuid.uuid4())
                curr_row[vl] = detection[id_col]
                annots.append(Row(**detection))
        matched_annots.append(annots)
        old_row = curr_row
    return matched_annots


annot_schema = T.ArrayType(
    T.StructType(
        [
            T.StructField("bbox", Box2dType(), False),
            T.StructField("tracker_id", T.StringType(), False),
        ]
    )
)


def track_detections(
    df, segment_id="vid", frames="frame", detections="detections"
):
    # utilities
    id_col = "tracker_id"
    frame_window = Window().orderBy(frames)
    value_window = Window().orderBy("value")
    annot_window = Window.partitionBy(segment_id).orderBy(segment_id, frames)
    indexer = StringIndexer(inputCol=segment_id, outputCol="vidIndex")

    # dataframe manipulation
    df = (
        df.select(segment_id, frames, detections)
        .withColumn("bbox", F.explode(detections))
        .withColumn(id_col, F.lit(""))
        .withColumn("trackables", struct([col("bbox"), col(id_col)]))
        .groupBy(segment_id, frames, detections)
        .agg(collect_list("trackables").alias("trackables"))
        .withColumn(
            "old_trackables", lag(col("trackables")).over(annot_window)
        )
        .withColumn(
            "matched", tracker_match(col("trackables"), col("old_trackables"))
        )
        .withColumn("frame_index", row_number().over(frame_window))
    )

    # update ids
    df = (
        indexer.fit(df)
        .transform(df)
        .withColumn("vidIndex", col("vidIndex").cast(StringType()))
    )
    unique_ids = df.select("vidIndex").distinct().count()
    matched = (
        df.select("vidIndex", segment_id, "trackables", "matched")
        .rdd.map(lambda x: (x[0], x[1:]))
        .partitionBy(unique_ids, lambda x: int(x[0]))
        .mapPartitions(match_annotations)
    )
    matched_annotations = spark.createDataFrame(
        matched, annot_schema
    ).withColumn("value_index", row_number().over(value_window))

    return (
        df.join(matched_annotations, col("value_index") == col("frame_index"))
        .withColumnRenamed("value", "trackers_matched")
        .withColumn("tracked", explode(col("trackers_matched")))
        .select(
            segment_id,
            frames,
            detections,
            col("tracked.{}".format("bbox")).alias("bbox"),
            col("tracked.{}".format(id_col)).alias(id_col),
        )
        .withColumn(id_col, sha2(concat(col(segment_id), col(id_col)), 256))
        .withColumn("tracked_detections", struct([col("bbox"), col(id_col)]))
        .groupBy(segment_id, frames, detections)
        .agg(F.collect_list("tracked_detections").alias("tracked_detections"))
        .orderBy(segment_id, frames, detections)
    )
