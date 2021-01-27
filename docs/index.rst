
Welcome to Rikai's documentation!
=================================

``Rikai`` is `Apache Parquet <https://parquet.apache.org/>`_ based format for unstructured Machine Learning (ML) dataset,
for example, video, image, or sensor data.

It offers **language and framework interoperable semantic types**,
and eliminates the tedious data conversions between the different stages in the ML life cycle.

These semantic types are natively supported in `Spark`_, `Jupyter`_, `Pytorch`_ and `Tensorflow`_.
For example, an :py:class:`~rikai.types.vision.Image` created from Spark will:

- Automatically converted into :py:class:`torch.Tensor` during model training if using :py:class:`rikai.torch.DataLoader`.
- Or appropriately presented via `Jupyter Display trait <https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html>`_
  in a Jupyter Notebook


Additionally, the parquet-native nature of the ``rikai`` format allows such unstructured ML dataset
being analyzed in Jupyter Notebook, `Spark`_, `Presto`_ or
`BigQuery <https://cloud.google.com/bigquery/external-data-cloud-storage>`_.

.. toctree::
   :maxdepth: 1

   quickstart
   types
   api/modules
   release

.. _Spark : https://spark.apache.org/
.. _Pytorch : https://pytorch.org/
.. _Tensorflow : https://www.tensorflow.org/
.. _Presto : https://prestodb.io/
.. _Jupyter : https://jupyter.org/

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

