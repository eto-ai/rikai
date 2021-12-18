#!/usr/bin/env python3
"""Create a test dataset from COCO datasets
"""

import argparse
import json
from itertools import islice

from pycocotools.coco import COCO


def build(args):
    coco = COCO(args.annotation_json)
    with open(args.annotation_json) as fobj:
        d = json.load(fobj)

    images = []
    annotations = []
    polygon_images = 0
    rle_images = 0
    for image_id in islice(coco.imgs, 0, 200):
        img = coco.loadImgs(image_id)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        if any([ann["iscrowd"] for ann in anns]):
            rle_images += 1
            if rle_images > 10:
                break
        else:
            polygon_images += 1
            if polygon_images > 10:
                continue
        images.extend(img)
        annotations.extend(anns)

    with open(args.output, "w") as fobj:
        json.dump({"annotations": annotations, "images": images}, fobj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_json", metavar="annotations.json")
    parser.add_argument("output", metavar="output.json")
    args = parser.parse_args()

    build(args)


if __name__ == "__main__":
    main()
