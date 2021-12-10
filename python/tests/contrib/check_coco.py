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
from typing import Tuple

import numpy as np
from pycocotools.coco import COCO

from rikai.types import rle
from rikai.types.geometry import Mask


def main():
    parser = argparse.ArgumentParser(
        description="Verify data consistency between coco and rikai"
    )
    parser.add_argument("annotation_json", help="Coco annotation json file")
    args = parser.parse_args()

    with open(args.annotation_json) as f:
        annotations = json.load(f)

    coco = COCO(args.annotation_json)

    IOU_THRESHOLD = 0.95

    for ann in annotations["annotations"]:
        img = coco.imgs[ann["image_id"]]
        height, width = img["height"], img["width"]  # type: int, int
        mask = Mask.from_mask(coco.annToMask(ann))

        if ann["iscrowd"] == 0:
            print(ann)
            mask_1 = Mask.from_polygon(ann["segmentation"], shape=(width, height))
        else:
            mask_1 = rle.decode(
                ann["segmentation"]["counts"], shape=(height, width, 1)
            ).reshape((height, width))

        iou = mask.iou(mask_1)
        print("IOU: ", mask.iou(mask_1))
        if iou < IOU_THRESHOLD:
            import cv2
            mask.to_mask()[mask == 1] = 255
            mask_1.to_mask()[mask_1 == 1] = 255
            # diff_mask[diff_mask == False] = 255
            cv2.imwrite("img1.png", mask)
            cv2.imwrite("img2.png", mask_1)
            # cv2.imwrite("diff.png", diff_mask)
            assert False, f"Image {ann['image_id']} not equal, {mask.shape}, {mask_1.shape}"


if __name__ == "__main__":
    main()
