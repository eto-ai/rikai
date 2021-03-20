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

from typing import Callable, Dict

from rikai.types.vision import Image

__all__ = ["fasterrcnn_resnet50_fpn"]


def _uri_to_pil(uri):
    # We can remove this after UDT is supported in Spark
    return Image(uri=uri).to_pil()


def _fasterrnn_post_processing(batch):
    results = []
    for predicts in batch:
        predict_result = {
            "boxes": [],
            "labels": [],
            "scores": [],
        }
        for box, label, score in zip(
            predicts["boxes"].tolist(),
            predicts["labels"].tolist(),
            predicts["scores"].tolist(),
        ):
            predict_result["boxes"].append(box)
            predict_result["labels"].append(label)
            predict_result["scores"].append(score)

        results.append(predict_result)
    return results


def fasterrcnn_resnet50_fpn() -> Dict[str, Callable]:
    """Pre/Post processing routines for
    :py:class:`torchvision.models.detection.fasterrcnn_resnet50_fpn`

    """
    from torchvision import transforms as T

    pre_processing = T.Compose(
        [
            _uri_to_pil,
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ]
    )
    return {
        "pre_processing": pre_processing,
        "post_processing": _fasterrnn_post_processing,
    }
