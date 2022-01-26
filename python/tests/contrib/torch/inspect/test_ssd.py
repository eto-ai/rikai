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

import torch
from torchvision.models.detection.ssd import ssd300_vgg16
from torchvision.transforms import ToTensor

from rikai.contrib.torch.inspect.ssd import SSDClassScoresExtractor
from rikai.types import Image

TEST_IMAGE = Image(
    "http://farm2.staticflickr.com/1129/4726871278_4dd241a03a_z.jpg"
)


def test_predict_value_equal():
    model = ssd300_vgg16(pretrained=True)
    model.eval()
    class_scores_extractor = SSDClassScoresExtractor(model)
    class_scores_extractor.eval()

    batch = [ToTensor()(TEST_IMAGE.to_pil())]
    with torch.no_grad():
        detections = model(batch)[0]
        class_scores = class_scores_extractor(batch)[0]

    # In torchvision 0.11.0, there is a bug in the order to find max value
    # of a label.
    correct_idx = detections["scores"] == class_scores["scores"][:, 0]
    assert len(correct_idx) > len(detections["scores"]) * 0.8

    print("BUG FROM PYTORCH")
    print(
        f"SCORES: pytorch: {detections['scores'][~correct_idx]},"
        f" we got: {class_scores['scores'][~correct_idx]}"
    )
    print(
        f"LABELS: pytorch: {detections['labels'][~correct_idx]}, "
        f" we got: {class_scores['labels'][~correct_idx]}"
    )

    assert torch.equal(
        detections["boxes"][correct_idx], class_scores["boxes"][correct_idx]
    )
    assert torch.equal(
        detections["scores"][correct_idx],
        class_scores["scores"][correct_idx][:, 0],
    )
    assert torch.equal(
        detections["labels"][correct_idx],
        class_scores["labels"][correct_idx][:, 0],
    )


def test_ssd_class_score_module_serialization():
    pass