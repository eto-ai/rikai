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
