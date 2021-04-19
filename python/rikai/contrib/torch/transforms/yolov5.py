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
"""
This is the implementation at https://pytorch.org/hub/ultralytics_yolov5/
With autoShape it does not require pre-processing
However it is NOT testable as it is not packaged as a python repository.
To use this make sure you install yolov5 and yolov5 requirements

```
pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
pip install yolov5
```
"""  # noqa E501

from typing import Any, Callable, Dict

__all__ = ["post_processing", "OUTPUT_SCHEMA"]


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
                "labels": [],
                "scores": [],
            }
            for *box, conf, cls in predicts.tolist():
                predict_result["boxes"].append(box)
                predict_result["labels"].append(cls)
                predict_result["scores"].append(conf)
        return results

    return post_process_func


OUTPUT_SCHEMA = (
    "struct<boxes:array<array<float>>, scores:array<float>, "
    "labels:array<int>>"
)
