#  Copyright 2022 Rikai Authors
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

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.detection.ssd import SSD
from torchvision.ops.boxes import batched_nms, clip_boxes_to_image

from rikai.pytorch.models.torchvision import ObjectDetectionModelType
from rikai.spark.sql.model import ModelSpec
from rikai.types import Box2d

__all__ = ["MODEL_TYPE"]


class SSDClassScoresExtractor(torch.nn.Module):
    """Extracts the scores (confidences) of each class for
    all the detected bounding box.

    Parameters
    ----------
    backend : :class:`torch.nn.Module`
        The trained SSD model
    topk_candidates : int, optional
        The number of top candidates (classes) returned per box.

    Returns
    -------
    [Dict[str, Tensor]]
        With the form of
    ``{"boxes": FloatTensor[N, 4], "labels": Int64Tensor[N, k], "scores": Tensor[N, k]}``
    """  # noqa: E501

    def __init__(self, backend: SSD, topk_candidates: int = 2):
        super().__init__()
        if not isinstance(backend, SSD):
            raise ValueError("Only support Torchvision's SSD model.")
        self.backend = backend
        self.topk_candidates = topk_candidates

    def forward(
        self,
        images: List[torch.Tensor],
    ):
        if self.training:
            raise ValueError("This feature extractor only supports eval mode.")

        # Reuse the code in torchvision's SSD.forward(), but returns multiple
        # classes and scores for each box.
        # TODO(GH-517): abstract the confidence extraction code for multiple
        #       torchvision models.

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, _ = self.backend.transform(images, None)

        # Get the features from the backbone
        features = self.backend.backbone(images.tensors)
        features = list(features.values())
        head_outputs = self.backend.head(features)
        pred_anchors = self.backend.anchor_generator(images, features)

        scores = head_outputs["cls_logits"]
        bbox_regression = head_outputs["bbox_regression"]
        pred_scores = F.softmax(scores, dim=-1)

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections: List[Dict[str, Tensor]] = []
        for boxes, scores, anchors, image_shape in zip(
            bbox_regression, pred_scores, pred_anchors, images.image_sizes
        ):
            boxes = self.backend.box_coder.decode_single(boxes, anchors)
            boxes = clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []
            image_all_scores = []
            for label in range(1, num_classes):
                score = scores[:, label]

                keep_idxs = score > self.backend.score_thresh
                score = score[keep_idxs]
                all_scores = scores[keep_idxs]
                box = boxes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = min(self.backend.topk_candidates, score.size(0))
                score, idxs = score.topk(num_topk)
                box = box[idxs]
                all_scores = all_scores[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_all_scores.append(all_scores)
                image_labels.append(
                    torch.full_like(
                        score,
                        fill_value=label,
                        dtype=torch.int64,
                        device=device,
                    )
                )

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            image_all_scores = torch.cat(image_all_scores, dim=0)

            # non-maximum suppression
            keep = batched_nms(
                image_boxes,
                image_scores,
                image_labels,
                self.backend.nms_thresh,
            )
            keep = keep[: self.backend.detections_per_img]
            top_scores, idx = image_all_scores[keep, 1:].topk(
                self.topk_candidates
            )
            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": top_scores,
                    "labels": idx + 1,
                }
            )
        detections = self.backend.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )
        return detections


class SSDClassScoresModelType(ObjectDetectionModelType):
    DEFAULT_MIN_SCORE = 0.3

    def __init__(self):
        super().__init__("SSDClassScores")

    def schema(self) -> str:
        return (
            "array<struct<box:box2d, scores:array<float>, "
            "label_ids:array<int>>>"
        )

    def load_model(self, spec: ModelSpec, **kwargs):
        model = spec.load_model()
        if isinstance(model, SSD):
            model = SSDClassScoresExtractor(model)
        model.eval()
        self.model = model
        if "device" in kwargs:
            self.model.to(kwargs.get("device"))
        self.spec = spec

    def predict(self, images, *args, **kwargs) -> List:
        assert (
            self.model is not None
        ), "model has not been initialized via load_model"
        min_score = float(
            self.spec.options.get("min_score", self.DEFAULT_MIN_SCORE)
        )

        batch = self.model(images)
        results = []
        for predicts in batch:
            predict_result = []
            for box, label, score in zip(
                predicts["boxes"].tolist(),
                predicts["labels"].tolist(),
                predicts["scores"].tolist(),
            ):
                if score[0] < min_score:
                    continue
                predict_result.append(
                    {
                        "box": Box2d(*box),
                        "label_id": label,
                        "score": score,
                    }
                )

            results.append(predict_result)
        return results


MODEL_TYPE = SSDClassScoresModelType()
