#!/usr/bin/env python3
#
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

"""Check Rikai is consistent with Coco python API.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO

from rikai.types import rle
from rikai.types.geometry import Mask


def main():
    parser = argparse.ArgumentParser(
        description="Verify data consistency between coco and rikai"
    )
    parser.add_argument("annotation_json", help="Coco annotation json file")
    parser.add_argument(
        "-i",
        "--iou",
        help="IOU threshold to consider a mask is mismatched",
        default=0.9,
        type=float,
    )
    args = parser.parse_args()

    with open(args.annotation_json) as f:
        annotations = json.load(f)

    coco = COCO(args.annotation_json)

    accuracy_record = []

    err_dir = Path("errors")
    shutil.rmtree(err_dir, ignore_errors=True)
    err_dir.mkdir()

    for ann in annotations["annotations"]:
        image_id = ann["image_id"]
        ann_id = ann["id"]
        img = coco.imgs[image_id]
        height, width = img["height"], img["width"]  # type: int, int
        is_rle = ann["iscrowd"] == 1
        # Mask from pycocotools directly
        original_mask = Mask.from_mask(coco.annToMask(ann))
        if np.count_nonzero(original_mask.to_mask()) == 0:
            continue

        if is_rle:
            rikai_mask = Mask.from_coco_rle(
                ann["segmentation"]["counts"], width=width, height=height
            )
        else:
            rikai_mask = Mask.from_polygon(
                ann["segmentation"],
                width=width,
                height=height,
            )

        iou = original_mask.iou(rikai_mask)
        accuracy_record.append(
            {
                "image_id": image_id,
                "iou": iou,
                "area": ann["area"],
                "rle": is_rle,
            }
        )
        print(f"Annotation: {ann_id} IOU: {original_mask.iou(rikai_mask)}")
        if iou < args.iou:
            coco_mask = original_mask.to_mask()
            r_mask = rikai_mask.to_mask()
            coco_mask[coco_mask == 1] = 255
            r_mask[r_mask == 1] = 255
            diff_mask = (coco_mask != r_mask).astype(np.uint8)
            diff_mask[diff_mask == 1] = 255
            cv2.imwrite(str(err_dir / f"{image_id}_coco.png"), coco_mask)
            cv2.imwrite(str(err_dir / f"{image_id}_rikai.png"), r_mask)
            cv2.imwrite(str(err_dir / f"{image_id}_diff.png"), diff_mask)
    df = pd.DataFrame(accuracy_record)
    df.to_csv("accuracy.csv")


if __name__ == "__main__":
    main()
