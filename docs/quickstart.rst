.. toctree::
   :maxdepth: 2

Quickstart
==========

In this quickstart, we illustrate a user journey from data cleaning to
model training to model evaluation using `PyTorch`_ and Rikai.

Installation
------------

.. code-block:: bash

    pip install rikai[torch]


Step 1. Feature Engineering
---------------------------

Let's start feature engineering using `Spark`_.

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
                "image": Image("s3://foo/bar/1.png"),
                "annotations": [
                    Row(
                        text="cat",
                        label=2,
                        mask=np.random(size=(256, 256)),
                        bbox=Box2d(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0)
                    )
                ]
            },
        ]
    )

    df.write.format("rikai").save("my_dataset")

The magic here is that Rikai maintains commonly used :doc:`Semantic Types <types>`
, takes care of ``SerDe`` and visualization in notebooks.

Additionally, Rikai community maintains a set of pre-baked connectors,
such as `COCO <https://cocodataset.org/#home>`_ and `ROS Bag <http://wiki.ros.org/Bags>`_

When it is ready, we can submit the script via ``spark-submit``

.. code-block:: bash

    spark-submit \
      --master yarn \
      script.py


Step 2. Inspect Dataset
------------------------

We can then inspect the dataset in a `Jupyter Notebook`_.

.. code-block:: python

    df = spark.read.format("rikai").load("my_dataset")
    df.printSchema()
    df.show(5)


Step 3. Train the model
-----------------------

Use this dataset in `Pytorch`_

.. code-block:: python

    import torch
    import torchvision
    from rikai.torch.vision import Dataset
    from torch.utils.data import DataLoader

    device = torch.device("cuda") if \
        torch.cuda.is_available() else torch.device("cpu")

    dataset = Dataset(
        "my_dataset",
        image_column="image",
        target_column='annotations',
        transform=torchvision.transforms.ToTensor(),
    )

    data_loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=4,
    )

    model.train()
    for epoch in range(10):
        for imgs, annotations in data_loader:
            loss_dict = model(imgs, annotations)
            losses = sum(loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

Rikai offers `MLflow`_ integration. When a model registered with `MLflow`, it will be available
to SQL ML directly.

.. code-block:: python

    import rikai.mlflow

    with mlflow.start_run() as run:
        # training loop
        for epoch in range(10):
            for imgs, annotations in data_loader:
                ...

        rikai.mlflow.pytorch.log_model(model, "model",
            model_type="ssd"
            registered_model_name="my_ssd")

Once the training finishes, Model ``my_ssd`` is available for :doc:`SQL ML <sqlml>` to use.

.. code-block:: SQL

    SELECT
        id,
        ML_PREDICT(my_ssd_model, image) as detections,
        annotations
    FROM my_dataset
    WHERE split = 'eval'
    LIMIT 10

.. _Spark : https://spark.apache.org/
.. _Jupyter Notebook : https://jupyter.org/
.. _Pytorch : https://pytorch.org/
.. _Mlflow : https://mlflow.org/


