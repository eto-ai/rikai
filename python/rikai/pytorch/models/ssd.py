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

import torchvision

from rikai.mixin import Pretrained
from rikai.pytorch.models.torchvision import ObjectDetectionModelType
from rikai.spark.sql.codegen.dummy import DummyModelSpec
from rikai.spark.sql.model import ModelSpec

__all__ = ["MODEL_TYPE"]


class SSDModelType(ObjectDetectionModelType, Pretrained):
    def pretrained_model(self):
        return torchvision.models.detection.ssd.ssd300_vgg16(pretrained=True)

    def find_model(self):
        if isinstance(self.spec, DummyModelSpec):
            return self.pretrained_model()
        else:
            return super().find_model()

    def __init__(self):
        super().__init__("SSD")


MODEL_TYPE = SSDModelType()
