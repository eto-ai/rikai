![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)

> :heavy_exclamation_mark: This repository is still experimental. No API-compatiblitly is guarenteed.

# rikai

Rikai is a `parquet` based feature store that has native Spark / Pytorch / Tensorflow support.
Additionally, it enables advanced analytic capabilties for model debuggabilty and monitoring.


## Example

```python
from pyspark.ml.linalg import DenseMetrix
from rikai.vision import Image, BBox
from rikai import numpy as np

df = spark.createDataFrame(
    [
        Row(
            id=1,
            mat=DenseMatrix(2, 2, range(4)),
            image=Image(uri="s3://foo/bar/1.png"),
            annotations=[
                Row(
                    label=Label("cat"),
                    mask=np.random(size=(256, 256)),
                    bbox=BBox(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
                ),
                Row(
                    label=Label("dog"),
                    mask=np.random(size=(256, 256)),
                    bbox=BBox(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
                ),
            ]
        ),
        Row(
            id=2,
            mat=DenseMatrix(3, 4, range(4)),
            image=Image(uri="s3://foo/bar/2.png"),
            annotations=[
                Row(
                    label=Label("car"),
                    mask=np.random(size=(256, 256)),
                    bbox=BBox(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
                ),
                Row(
                    label=Label("flag"),
                    mask=np.random(size=(256, 256)),
                    bbox=BBox(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
                ),
            ]
        ),
    ]
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
