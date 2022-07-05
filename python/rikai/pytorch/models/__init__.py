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

"""Rikai-implemented PyTorch models and executors."""

import importlib

torchvision_found = importlib.util.find_spec("torchvision") is not None

if torchvision_found:
    import rikai.pytorch.models.convnext
    import rikai.pytorch.models.efficientnet
    import rikai.pytorch.models.fasterrcnn
    import rikai.pytorch.models.feature_extractor
    import rikai.pytorch.models.keypointrcnn
    import rikai.pytorch.models.maskrcnn
    import rikai.pytorch.models.resnet
    import rikai.pytorch.models.retinanet
    import rikai.pytorch.models.ssd
    import rikai.pytorch.models.ssd_class_scores

from rikai.pytorch.models.torch import MODEL_TYPES  # noqa

__all__ = ["MODEL_TYPES"]
