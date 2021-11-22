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

from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from yolov5.utils.datasets import exif_transpose

from rikai.types.vision import Image

__all__ = ["pre_processing", "post_processing", "OUTPUT_SCHEMA"]


def pre_process_func(im):
    im = Image(im).to_pil()
    im = np.asarray(exif_transpose(im))
    if im.shape[0] < 5:  # image in CHW
        im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
    im = (
        im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)
    )  # enforce 3ch input
    return torch.from_numpy(im)


def pre_processing(options: Dict[str, Any]) -> Callable:
    return pre_process_func


def post_processing(options: Dict[str, Any]) -> Callable:
    def post_process_func(batch: "Detections"):
        """
        Parameters
        ----------
        batch: Detections
            The ultralytics yolov5 (in torch hub) autoShape output
        """
        results = []
        for predicts in batch.pred:
            predict_result = {
                "boxes": [],
                "label_ids": [],
                "scores": [],
            }
            for *box, conf, cls in predicts.tolist():
                predict_result["boxes"].append(box)
                predict_result["label_ids"].append(cls)
                predict_result["scores"].append(conf)
            results.append(predict_result)
        return results

    return post_process_func


OUTPUT_SCHEMA = (
    "struct<boxes:array<array<float>>, scores:array<float>, "
    "label_ids:array<int>>"
)
