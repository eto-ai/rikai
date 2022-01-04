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

"""
Adapted from https://github.com/fcakyon/yolov5-pip/blob/5.0.10/yolov5/models/common.py#L291

For the Yolov5 Autoshape Module, preprocessing/postprocessing are mixed: the
tensors of the prepared images during preprocessing are also used in
postprocessing. That's why we need to customize the UDF generator and the
post_processing parameters.

Another approach is that we can create a rikai-friendly module wrapper for the
yolov5 model. It should work for most modules, but for torchscript model, it
might not work.
"""  # noqa E501

from typing import Any, Callable, Dict

import numpy as np
import torch
from torch.cuda import amp
from yolov5.models.common import Detections
from yolov5.utils.datasets import exif_transpose, letterbox
from yolov5.utils.general import (
    make_divisible,
    non_max_suppression,
    scale_coords,
)
from yolov5.utils.torch_utils import time_sync

from rikai.types import Image

__all__ = ["pre_processing", "post_processing", "OUTPUT_SCHEMA"]


def pre_process_func(im: Image):
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
    augment = bool(options.get("augment", False))
    profile = bool(options.get("profile", False))
    # NMS confidence threshold
    conf_thres = float(options.get("conf_thres", 0.25))
    # NMS IoU threshold
    iou_thres = float(options.get("iou_thres", 0.45))
    # maximum number of detections per image
    max_det = int(options.get("max_det", 1000))
    image_size = int(options.get("image_size", 640))

    def post_process_func(model, batch):
        t = [time_sync()]

        p = next(model.parameters())  # for device and type
        n = len(batch)
        imgs = []
        for item in batch:
            imgs.append(item.cpu().numpy())
        shape0 = []
        shape1 = []
        for i, im in enumerate(imgs):
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = image_size / max(s)  # gain
            shape1.append([y * g for y in s])
            imgs[i] = (
                im if im.data.contiguous else np.ascontiguousarray(im)
            )  # update
        shape1 = [
            make_divisible(x, int(model.stride.max()))
            for x in np.stack(shape1, 0).max(0)
        ]  # inference shape
        x = [
            letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs
        ]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = (
            torch.from_numpy(x).to(p.device).type_as(p) / 255.0
        )  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=p.device.type != "cpu"):
            pred = model(x.to(p.device).type_as(p), augment, profile)
            y = pred[0]
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(
                y, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det
            )  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])
            t.append(time_sync())

            detections = Detections(imgs, y, None, times=t, shape=x.shape)

            results = []
            for predicts in detections.pred:
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
