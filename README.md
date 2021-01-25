![Apache License](https://img.shields.io/github/license/eto-ai/rikai?style=for-the-badge)
[![Read The Doc](https://img.shields.io/readthedocs/rikai?style=for-the-badge)](https://rikai.readthedocs.io/)
![Github Action](https://img.shields.io/github/workflow/status/eto-ai/rikai/Python?style=for-the-badge)
![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg?style=for-the-badge)

> :heavy_exclamation_mark: This repository is still experimental. No API-compatibility is guaranteed.

# rikai

Rikai is a [`parquet`](https://parquet.apache.org/) based ML data format built for working with
unstructured data at scale. Processing large amounts of data for ML is never trivial, but that
is especially true for images and videos often at the core of deep learning applications. We are
building Rikai with two main goals:
1. Enable ML engineers/researchers to have a seamless workflow from Spark to PyTorch/TF, 
   from notebook to production.
2. Enable advanced analytics capabilities to support much faster active learning, model debugging,
   and monitoring in production pipelines.

Current (v0.0.1) main features:
1. Native support in Spark and PyTorch for images/videos: reduce ad-hoc type
   conversions when moving between ETL and training.
2. Custom functionality for working with images and videos at scale: reduce boilerplate and 
   low-level code currently required to process images, filter/sample videos, etc.

Roadmap:
1. TensorFlow integration
2. Versioning support built into the dataset
3. Richer video capabilities (ffmpeg-python integration)
4. Declarative annotation API (think vega-lite for annotating images/videos)
5. Data-centric analytics API (think BigQuery ML)

## Example

```python
from pyspark.ml.linalg import DenseMetrix
from rikai.vision import Image, BBox
from rikai import numpy as np

df = spark.createDataFrame(
    [{
        "id": 1,
        "mat": DenseMatrix(2, 2, range(4)),
        "image": Image("s3://foo/bar/1.png"),
        "annotations": [
            {
                "label": Label("cat"),
                "mask": np.random(size=(256,256)),
                "bbox": BBox(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0)
            }
        ]
    }]
)

df.write.format("rikai").save("s3://path/to/features")
```

Train dataset in `pytorch`

```python
from rikai.torch import DataLoader

data_loader = DataLoader(
    "s3://foo/bar",
    batch_size=32,
    shuffle=True,
    num_workers=8,
)
for example in data_loader:
    print(example)
```


## Docker

The included Dockerfile creates a standalone demo image with
Jupyter, Pytorch, Spark, and rikai preinstalled with notebooks for you
to play with the capabilities of the rikai feature store.

### Building the image

1. First build the rikai jar

``` bash
# from rikai root
sbt publishLocal # assumes Scala 2.12 and version is 0.0.1
```

``` bash
# from rikai root
# if the base system is linux and you want to expose the gpu, ensure `nvidia-docker2` is installed first
docker build --tag rikai --network host .
```

2. Running the image

``` bash
docker run -p 0.0.0.0:8888:8888/tcp rikai:latest jupyter lab --ip 0.0.0.0 --port 8888 --NotebookApp.quit_button=True --NotebookApp.custom_display_url=http://127.0.0.1:8888
```

The console should print out a clickable link to JupyterLab. Run the provided notebook to validate that spark standalone, pytorch, and rikai are installed correctly.
