.. toctree::
   :maxdepth: 1

Quickstart
==========

In this quickstart, we illustrate a user journey from data cleaning to
model training to model evaluation using `Pytorch`_ and Rikai.

Installation
------------

.. code-block:: bash

    pip install rikai[torch]


Use `Rikai` in the machine learning life cycle
----------------------------------------------

Lets get started from the feature engineering in `Spark`_.


.. code-block:: python

    from pyspark.sql import SparkSession
    from pyspark.ml.linalg import DenseMetrix
    from rikai.types import Image, Box2d
    from rikai import numpy as np

    spark = (
        SparkSession
        .builder
        .appName("rikai-quickstart")
        .getOrCreate()
    )

    df = spark.createDataFrame(
        [
            {
                "id": 1,
                "mat": DenseMatrix(2, 2, range(4)),
                "image": Image("s3://foo/bar/1.png"),
                "annotations": [
                    Row(
                        text="cat",
                        mask=np.random(size=(256, 256)),
                        bbox=Box2d(x=1.0, y=2.0, width=3.0, height=4.0)
                    )
                ]
            },
        ]
    )

    df.write.format("rikai").save("dataset/out")

We can then inspect the dataset in a `Jupyter Notebook`_.



.. code-block:: python

    df = spark.read.format("rikai").load("dataset/out")
    df.printSchema()
    df.show(5)



Use the dataset in `pytorch`

.. code-block:: python

    from rikai.torch import DataLoader

    data_loader = DataLoader(
        "dataset/out",
        shuffle=True,
        batch=8,
    )
    for examples in data_loader:
        print(example)

.. _Spark : https://spark.apache.org/
.. _Jupyter Notebook : https://jupyter.org/
.. _Pytorch : https://pytorch.org/


