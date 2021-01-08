.. Rikai documentation master file, created by
   sphinx-quickstart on Thu Jan  7 22:46:51 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Rikai's documentation!
=================================

``Rikai`` is a feature store for unstructured data, i.e., video, image, or sensor data.

.. code-block:: python

   from pyspark.ml.linalg import DenseMetrix
   from rikai.vision import Image, BBox
   from rikai import numpy as np

   df = spark.createDataFrame(
      [
         {
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
         },
      ]
   )

   df.write.format("rikai").save("s3://path/to/features")


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
