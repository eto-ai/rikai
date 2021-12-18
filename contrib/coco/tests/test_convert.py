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

from pathlib import Path

from pycocotools.coco import COCO
import numpy as np

from rikai.types import Mask


def test_covert_segmentation():
    coco = COCO(str(Path(__file__).parent / "testdata" / "coco.json"))

    for image_id in coco.imgs:
        img = coco.loadImgs(image_id)[0]
        # print(img)
        height, width = img["height"], img["width"]
        ann_ids = coco.getAnnIds(imgIds=image_id)
        for ann in coco.loadAnns(ann_ids):
            if ann["iscrowd"] == 0:
                mask = Mask.from_polygon(
                    ann["segmentation"], height=height, width=width
                )
                # TODO: currently, it has disparity of the conversion between
                # polygon to mask between coco and rikai (PIL based)
            else:
                mask = Mask.from_coco_rle(
                    ann["segmentation"]["counts"], height=height, width=width
                )
                assert np.array_equal(mask.to_mask(), coco.annToMask(ann))
