![Apache License](https://img.shields.io/github/license/eto-ai/rikai?style=for-the-badge)
[![Read The Doc](https://img.shields.io/readthedocs/rikai?style=for-the-badge)](https://rikai.readthedocs.io/)
[![javadoc](https://javadoc.io/badge2/ai.eto/rikai_2.12/javadoc.svg?style=for-the-badge)](https://javadoc.io/doc/ai.eto/rikai_2.12)
![Pypi version](https://img.shields.io/pypi/v/rikai?style=for-the-badge)
![Github Action](https://img.shields.io/github/workflow/status/eto-ai/rikai/Python?style=for-the-badge)
![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg?style=for-the-badge)


Join the community:
[![Join the chat at https://gitter.im/rikaidev/community](https://img.shields.io/badge/chat-on%20gitter-green?style=for-the-badge)](https://gitter.im/rikaidev/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

> :heavy_exclamation_mark: This repository is still experimental. No API-compatibility is guaranteed.

# Rikai

Rikai is the easiest way to work with computer vision datasets at scale for data prep, model deployment, and
model inferencing that's integrated with your data warehouse. 

If you have docker, you can easily try out the Rikai quickstart docker image by cloning this repo and then running:

```bash
docker compose up quickstart
```

It should print out a link to jupyter lab server. Once opened you'll find quickstart notebooks that you can step 
through.

## Main Features

### Data format

Rikai adds semantic types for computer vision on [Apache Parquet](https://parquet.apache.org/) like Image, Box2d,
Mask, and more.

### SQL-ML Engine

Rikai extends Spark SQL with ML capability which allows users to analyze Rikai datasets using own models with SQL
("Bring your own model")

### Visualization

Carefully crafted data-visualization embedded with semantic types, especially in Jupyter notebooks,
to help you visualize and inspect your AI data without having to remember complicated raw image manipulations.

### Integrations

Rikai comes with an extensive set of I/O connectors. For ETL, Rikai is able to consume popular formats like ROS bags and Coco.
For analysis, it's easy to read Rikai data into pandas/spark DataFrames (Rikai handles serde for the semantic types).
And for training, Rikai allows direct creation of Pytorch/Tensorflow datasets without manual conversion.


## Example

```python
from pyspark.sql import Row
from pyspark.ml.linalg import DenseMatrix
from rikai.types import Image, Box2d
from rikai.numpy import wrap
import numpy as np

df = spark.createDataFrame(
    [
        {
            "id": 1,
            "mat": DenseMatrix(2, 2, range(4)),
            "image": Image("s3://foo/bar/1.png"),
            "annotations": [
                Row(
                    label="cat",
                    mask=wrap(np.random.rand(256, 256)),
                    bbox=Box2d(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
                )
            ],
        }
    ]
)

df.write.format("rikai").save("s3://path/to/features")
```

Train dataset in `Pytorch`

```python
from torch.utils.data import DataLoader
from torchvision import transforms as T
from rikai.pytorch.vision import Dataset

transform = T.Compose([
   T.Resize(640),
   T.ToTensor(),
   T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset = Dataset(
   "s3://path/to/features",
   image_column="image",
   transform=transform
)
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
)
for batch in loader:
    predicts = model(batch.to("cuda"))
```

Using a ML model in Spark SQL (**experiemental**)

```sql
CREATE MODEL yolo5
OPTIONS (min_confidence=0.3, device="gpu", batch_size=32)
USING "s3://bucket/to/yolo5_spec.yaml";

SELECT id, ML_PREDICT(yolo5, image) FROM my_dataset
WHERE split = "train" LIMIT 100;
```

Rikai can use MLflow as its model registry. This allows you to automatically pickup the latest
model version if you're using the mlflow model registry. Here is a list of supported model flavors:
+ PyTorch (pytorch)
+ Tensorflow (tensorflow)
+ Scikit-learn (sklearn)

```sql
CREATE MODEL yolo5
OPTIONS (min_confidence=0.3, device="gpu", batch_size=32)
USING "mlflow:///yolo5_model/";

SELECT id, ML_PREDICT(yolo5, image) FROM my_dataset
WHERE split = "train" LIMIT 100;
```

For more details on the model spec, see [SQL-ML documentation](https://rikai.readthedocs.io/en/latest/sqlml.html)

## Getting Started

Currently Rikai is maintained for <a name="VersionMatrix"></a>Scala 2.12 and Python 3.7, 3.8, 3.9

There are multiple ways to install Rikai:

1. Try it using the included docker image (`docker compose up`).
2. Install via pip `pip install rikai` (with
   [extras for gcp, pytorch/tf, and others](#Extras)).
3. Install from [source](#Source)

Note: if you want to use Rikai with your own pyspark, please consult
[rikai documentation](https://rikai.readthedocs.io/en/latest/spark.html) for tips.

### <a name="Extras"></a>Install from pypi

Base rikai library can be installed with just `pip install rikai`. Dependencies for supporting
pytorch (pytorch and torchvision), jupyter (matplotlib and jupyterlab) are all part of
optional extras. Many open-source datasets also use Youtube videos so we've also added pafy and
youtube-dl as optional extras as well.

For example, if you want to use pytorch in Jupyter to train models on rikai datasets in s3
containing Youtube videos you would run:

`pip install rikai[pytorch,jupyter,youtube]`

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

### Utilities

[pre-commit](https://pre-commit.com/) can be helpful in keep consistent code format with the repository. 
It can trigger reformat and extra things in your local machine before the CI force you to do it.

If you want it, install and enable `pre-commit`
```bash
pip install pre-commit
pre-commit install #in your local development directory
#pre-commit installed at .git/hooks/pre-commit
```
If you want to uninstall it, it would be easy, too.
```
pre-commit uninstall
```
