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

"""Convert Coco dataset into Rikai format.

https://cocodataset.org/#home
"""

from __future__ import annotations

import json
import os
from itertools import islice
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

try:
    from pycocotools.coco import COCO
except ImportError as exc:
    raise ImportError("Please install pycocotools") from exc

from rikai.spark.functions import image_copy
from rikai.spark.types import Box2dType, ImageType
from rikai.types import Box2d, Image

__all__ = ["convert"]


def load_categories(annotation_file):
    """Load categories to be global categories."""
    with open(annotation_file) as fobj:
        ann = json.load(fobj)
        return {c["id"]: c for c in ann["categories"]}


def convert(
    spark: SparkSession,
    dataset_root: str,
    limit: int = 0,
    asset_dir: Optional[str] = None,
) -> DataFrame:
    """Convert a Coco Dataset into Rikai dataset.

    This function expects the COCO datasets are stored in directory with the
    following structure:

    - dataset
        - annotations
          - captions_train2017.json
          - instances_train2017.json
          - ...
        - train2017
        - val2017
        - test2017

    Parameters
    ----------
    spark : SparkSession
        A live spark session
    dataset_root : str
        The directory of dataset
    limit : int, optional
        The number of images of each split to be converted.
    asset_dir : str, optional
        The asset directory to store images, can be a s3 directory.

    Return
    ------
    DataFrame
        Returns a Spark DataFrame
    """
    train_json = os.path.join(
        dataset_root, "annotations", "instances_train2017.json"
    )
    val_json = os.path.join(
        dataset_root, "annotations", "instances_val2017.json"
    )

    categories = load_categories(train_json)

    examples = []
    for split, anno_file in zip(["train", "val"], [train_json, val_json]):
        coco = COCO(annotation_file=anno_file)
        # Coco has native dependencies, so we do not distributed them
        # to the workers.
        image_ids = coco.imgs
        if limit > 0:
            image_ids = islice(image_ids, limit)
        for image_id in image_ids:
            ann_id = coco.getAnnIds(imgIds=image_id)
            annotations = coco.loadAnns(ann_id)
            annos = []
            for ann in annotations:
                bbox = Box2d.from_top_left(*ann["bbox"])
                annos.append(
                    {
                        "category_id": ann["category_id"],
                        "category_text": categories[ann["category_id"]][
                            "name"
                        ],
                        "bbox": bbox,
                        "area": float(ann["area"]),
                    }
                )
            image_payload = coco.loadImgs(ids=image_id)[0]
            example = {
                "image_id": image_id,
                "annotations": annos,
                "image": Image(
                    os.path.abspath(
                        os.path.join(
                            dataset_root,
                            "{}2017".format(split),
                            image_payload["file_name"],
                        )
                    )
                ),
                "split": split,
            }
            examples.append(example)

    schema = StructType(
        [
            StructField("image_id", LongType(), False),
            StructField(
                "annotations",
                ArrayType(
                    StructType(
                        [
                            StructField("category_id", IntegerType()),
                            StructField("category_text", StringType()),
                            StructField("area", FloatType()),
                            StructField("bbox", Box2dType()),
                        ]
                    )
                ),
                False,
            ),
            StructField("image", ImageType(), False),
            StructField("split", StringType(), False),
        ]
    )
    df = spark.createDataFrame(examples, schema=schema)
    if asset_dir:
        asset_dir = asset_dir if asset_dir.endswith("/") else asset_dir + "/"
        print("ASSET DIR: ", asset_dir)
        df = df.withColumn("image", image_copy(col("image"), lit(asset_dir)))
    return df
