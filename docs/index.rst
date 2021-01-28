
Welcome to Rikai's documentation!
=================================

Rikai is an `Apache Spark`_ based ML data format built for working with
unstructured data at scale. Processing large amounts of data for ML is never trivial, but that
is especially true for images and videos often at the core of deep learning applications. We are
building Rikai with two main goals:

1. Enable ML engineers/researchers to have a seamless workflow from feature engineering (`Spark`_)
   to training (`PyTorch`_/`Tensorflow`_), from notebook to production.
2. Enable advanced analytics capabilities to support much faster active learning, model debugging,
   and monitoring in production.


It offers **language and framework interoperable semantic types**,
and eliminates the tedious data conversions between the different stages in the ML life cycle.

.. code-block:: python

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

  df.write.format("rikai").save("dataset/out")

``Rikai`` dataset can be seamlessly integrated into your favorite training frameworks,
taking `Pytorch`_ as an example:

.. code-block:: python

  from rikai.torch import DataLoader

  data_loader = DataLoader(
      "dataset/out",
      shuffle=True,
      batch=8,
  )
  for examples in data_loader:
      print(example)

Additionally, the parquet-native nature of the ``rikai`` format allows such unstructured ML dataset
being analyzed in `Jupyter`_, `Spark`_, `Presto`_ or
`BigQuery <https://cloud.google.com/bigquery/external-data-cloud-storage>`_.

For more details, please read :doc:`quickstart`.


.. toctree::
   :maxdepth: 1

   quickstart
   types
   api/modules


.. _Spark : https://spark.apache.org/
.. _Pytorch : https://pytorch.org/
.. _Tensorflow : https://www.tensorflow.org/
.. _Presto : https://prestodb.io/
.. _Jupyter : https://jupyter.org/
.. _Apache Spark : https://parquet.apache.org/

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

