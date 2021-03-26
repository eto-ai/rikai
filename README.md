![Apache License](https://img.shields.io/github/license/eto-ai/rikai?style=for-the-badge)
[![Read The Doc](https://img.shields.io/readthedocs/rikai?style=for-the-badge)](https://rikai.readthedocs.io/)
[![javadoc](https://javadoc.io/badge2/ai.eto/rikai_2.12/javadoc.svg?style=for-the-badge)](https://javadoc.io/doc/ai.eto/rikai_2.12)
[![Join the chat at https://gitter.im/rikaidev/community](https://img.shields.io/badge/chat-on%20gitter-green?style=for-the-badge)](https://gitter.im/rikaidev/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
![Pypi version](https://img.shields.io/pypi/v/rikai?style=for-the-badge)
![Github Action](https://img.shields.io/github/workflow/status/eto-ai/rikai/Python?style=for-the-badge)
![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg?style=for-the-badge)

> :heavy_exclamation_mark: This repository is still experimental. No API-compatibility is guaranteed.

# Rikai

Rikai is a [`parquet`](https://parquet.apache.org/) based ML data format built for working with
unstructured data at scale. Processing large amounts of data for ML is never trivial, but that
is especially true for images and videos often at the core of deep learning applications. We are
building Rikai with two main goals:
1. Enable ML engineers/researchers to have a seamless workflow from Feature Engineering (Spark) to Training (PyTorch/Tensorflow),
   from notebook to production.
2. Enable advanced analytics capabilities to support much faster active learning, model debugging,
   and monitoring in production pipelines.

Current (v0.0.1) main features:
1. Native support in Spark and PyTorch for images/videos: reduce ad-hoc type
   conversions when moving between ETL and training.
2. Custom functionality for working with images and videos at scale: reduce boilerplate and
   low-level code currently required to process images, filter/sample videos, etc.
3. ML-enabled SQL analytics API

Roadmap:
1. TensorFlow integration
2. Versioning support built into the dataset
3. Richer video capabilities (ffmpeg-python integration)
4. Declarative annotation API (think vega-lite for annotating images/videos)
5. Data-centric analytics API (think BigQuery ML)

## Example

```python
from pyspark.ml.linalg import DenseMetrix
from rikai.types import Image, Box2d
from rikai import numpy as np

df = spark.createDataFrame(
    [{
        "id": 1,
        "mat": DenseMatrix(2, 2, range(4)),
        "image": Image("s3://foo/bar/1.png"),
        "annotations": [
            {
                "label": "cat",
                "mask": np.random(size=(256,256)),
                "bbox": Box2d(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0)
            }
        ]
    }]
)

df.write.format("rikai").save("s3://path/to/features")
```

Train dataset in `Pytorch`

```python
from rikai.torch import DataLoader

data_loader = DataLoader(
    "s3://path/to/features",
    batch_size=32,
    shuffle=True,
    num_workers=8,
)
for example in data_loader:
    print(example)
```

## Getting Started

Currently Rikai is maintained for <a name="VersionMatrix"></a>Scala 2.12 and Python 3.7 and 3.8.

There are multiple ways to install Rikai:

1. Try it using the included [Dockerfile](#Docker).
2. OR install it via pip `pip install rikai`, with
   [extras for aws/gc, pytorch/tf, and others](#Extras).
3. OR install it from [source](#Source)

Note: if you want to use Rikai with your own pyspark, please consult rikai documentation for tips.

### <a name="Docker"></a>Docker

The included Dockerfile creates a standalone demo image with
Jupyter, Pytorch, Spark, and rikai preinstalled with notebooks for you
to play with the capabilities of the rikai feature store.

To build and run the docker image from the current directory:
```bash
# Clone the repo
git clone git@github.com:eto-ai/rikai rikai
# Build the docker image
docker build --tag rikai --network host .
# Run the image
docker run -p 0.0.0.0:8888:8888/tcp rikai:latest jupyter lab -ip 0.0.0.0 --port 8888
```

If successful, the console should then print out a clickable link to JupyterLab. You can also
open a browser tab and go to `localhost:8888`.

### <a name="Extras"></a>Install from pypi

Base rikai library can be installed with just `pip install rikai`. Dependencies for supporting
pytorch (pytorch and torchvision), aws (boto), jupyter (matplotlib and jupyterlab) are all part of
optional extras. Many open-source datasets also use Youtube videos so we've also added pafy and
youtube-dl as optional extras as well.

For example, if you want to use pytorch in Jupyter to train models on rikai datasets in s3
containing Youtube videos you would run:

`pip install rikai[pytorch,aws,jupyter,youtube]`

If you're not sure what you need and don't mind installing some extra dependencies, you can
simply install everything:

`pip install rikai[all]`

### <a name="Source"></a>Install from source

To build from source you'll need python as well as Scala with sbt installed:

```bash
# Clone the repo
git clone git@github.com:eto-ai/rikai rikai
# Build the jar
sbt publishLocal
# Install python package
cd python
pip install -e . # pip install -e .[all] to install all optional extras (see "Install from pypi")
```
