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

"""
Adapted from https://github.com/pytorch/pytorch.github.io/blob/site/assets/hub/pytorch_vision_resnet.ipynb
"""  # noqa E501


def pre_processing(options: Dict[str, Any]) -> Callable:
    return T.Compose(
        [
            uri_to_pil,
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def post_processing(options: Dict[str, Any]) -> Callable:
    def post_process_func(batch):
        results = []
        for result in batch:
            results.append(result.detach().cpu().numpy())
        return results

    return post_process_func


OUTPUT_SCHEMA = "array<float>"