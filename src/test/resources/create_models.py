#!/usr/bin/env python3
#
#  Copyright (c) 2022 Rikai Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Register models into Mlflow for scala tests."""

import argparse

import mlflow
import torch
import torchvision

import rikai


def register_torch_model(model: torch.nn.Module, name: str):
    artifact_path = "model"
    rikai.mlflow.pytorch.log_model(
        model,
        artifact_path,
        model_type="resnet",
        registered_model_name=name,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-uri", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    with mlflow.start_run(run_id=args.run_id):
        resnet = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            progress=False,
        )
        register_torch_model(resnet, "resnet")
        ssd = torchvision.models.detection.ssd300_vgg16(
            pretrained=True,
            progress=False,
        )
        register_torch_model(ssd, "ssd")


if __name__ == "__main__":
    main()
