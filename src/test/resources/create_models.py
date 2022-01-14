#!/usr/bin/env python3

"""Register models into Mlflow for scala tests."""

import argparse

import mlflow
import torch
import torchvision

import rikai
from rikai.contrib.torch.detections import OUTPUT_SCHEMA


def register_torch_model(model: torch.nn.Module, name: str):
    artifact_path = "model"
    pre_processing = (
        "rikai.contrib.torch.transforms."
        "fasterrcnn_resnet50_fpn.pre_processing"
    )
    post_processing = (
        "rikai.contrib.torch.transforms."
        "fasterrcnn_resnet50_fpn.post_processing"
    )
    rikai.mlflow.pytorch.log_model(
        model,
        artifact_path,
        OUTPUT_SCHEMA,
        pre_processing,
        post_processing,
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
