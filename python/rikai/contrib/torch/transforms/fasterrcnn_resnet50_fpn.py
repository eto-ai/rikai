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

from typing import Any, Callable, Dict

from torchvision import transforms as T

from rikai.contrib.torch.transforms.utils import uri_to_pil

__all__ = ["pre_processing", "post_processing"]

DEFAULT_MIN_SCORE = 0.5


def pre_processing(options: Dict[str, Any]) -> Callable:
    return T.Compose(
        [
            uri_to_pil,
            T.ToTensor(),
        ]
    )


def post_processing(options: Dict[str, Any]) -> Callable:
    min_score = float(options.get("min_score", DEFAULT_MIN_SCORE))

    def post_process_func(batch):
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
                if score < min_score:
                    continue
                predict_result["boxes"].append(box)
                predict_result["labels"].append(label)
                predict_result["scores"].append(score)

            results.append(predict_result)
        return results

    return post_process_func


OUTPUT_SCHEMA = (
    "struct<boxes:array<array<float>>, scores:array<float>, "
    "labels:array<int>>"
)
