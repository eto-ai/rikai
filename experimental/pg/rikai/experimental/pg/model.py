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

from typing import Optional

import torch

from rikai.spark.sql.codegen.dummy import DummyModelSpec
from rikai.spark.sql.codegen.fs import FileModelSpec
from rikai.spark.sql.model import ModelType
from rikai.types import Image


class PgModel:
    def __init__(self, model_type: ModelType):
        self.model = model_type

    def __repr__(self):
        return f"PgModel({self.model})"

    def predict(self, img):
        tensor = torch.tensor(
            self.model.transform()(Image(img["uri"]).to_numpy())
        )
        preds = self.model.predict([tensor])[0]

        return [
            {
                "label": pred["label"],
                "label_id": pred["label_id"],
                "score": pred["score"],
                "box": (
                    (pred["box"].xmin, pred["box"].ymin),
                    (pred["box"].xmax, pred["box"].ymax),
                ),
            }
            for pred in preds
        ]


def load_model(
    flavor: str,
    model_type: str,
    uri: Optional[str] = None,
    options: Optional[dict[str, str]] = None,
) -> PgModel:
    # TODO: move load model into rikai core.
    conf = {
        "version": "1.0",
        "name": f"load_{model_type}",
        "flavor": flavor,
        "modelType": model_type,
        "uri": uri,
    }
    if uri:
        spec = FileModelSpec(conf)
    else:
        spec = DummyModelSpec(conf)
    model = spec.model_type
    model.load_model(spec)
    return PgModel(model)
