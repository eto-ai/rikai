#!/usr/bin/env python3

"""Creating models to Mlflow for tests.

"""

import torch
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-uri", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()


if __name__ == "__main__":
    main()
