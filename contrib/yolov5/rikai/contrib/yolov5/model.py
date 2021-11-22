import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
from yolov5.models.common import Detections
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import (
    make_divisible,
    non_max_suppression,
    scale_coords,
)
from yolov5.utils.torch_utils import time_sync

from rikai.contrib.torch import RikaiModule


class RikaiYolov5Model(nn.Module, RikaiModule):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    @torch.no_grad()
    def forward(self, options, batch):
        t = [time_sync()]

        augment = options.get("augment", False)
        profile = options.get("profile", False)
        conf_thres = options.get(
            "conf_thres", 0.25
        )  # NMS confidence threshold
        iou_thres = options.get("iou_thres", 0.45)  # NMS IoU threshold
        max_det = options.get(
            "max_det", 1000
        )  # maximum number of detections per image
        image_size = options.get("image_size", 640)

        p = next(self.model.parameters())  # for device and type
        n = len(batch)
        imgs = []
        for item in batch:
            imgs.append(item.numpy())
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
            make_divisible(x, int(self.model.stride.max()))
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
            pred = self.model(x.to(p.device).type_as(p), augment, profile)
            y = pred[0]
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(
                y, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det
            )  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])
            t.append(time_sync())

            return Detections(imgs, y, None, times=t, shape=x.shape)
