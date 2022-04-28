
Welcome to Rikai!
*****************

Rikai is a framework specifically designed for AI workflows focused around large scale unstructured datasets
(e.g., images, videos, sensor data (future), text (future), and more).
Through every stage of the AI modeling workflow,
Rikai strives to offer a great developer experience when working with real-world AI datasets.

The quality of an AI dataset can make or break an AI project, but tooling for AI data is sorely lacking in ergonomics.
As a result, practitioners must spend most of their time and effort wrestling with their data instead of innovating on the models and use cases.
Rikai alleviates the pain that AI practitioners experience on a daily basis dealing with the myriad of tedious data tasks,
so they can focus again on model-building and problem solving.


Main Features
=============

Data format
^^^^^^^^^^^

The core of Rikai is a data format ("rikai format")
based on `Apache Parquet`_.
Rikai augments parquet with a rich collection of semantic types design specifically for unstructured data and annotations.

Integrations
^^^^^^^^^^^^

Rikai comes with an extensive set of I/O connectors. For ETL, Rikai is able to consume popular formats like ROS bags and Coco.
For analysis, it's easy to read Rikai data into pandas/spark DataFrames (Rikai handles serde for the semantic types).
And for training, Rikai allows direct creation of Pytorch/Tensorflow datasets without manual conversion.

SQL-ML Engine
^^^^^^^^^^^^^

Rikai extends Spark SQL with ML capability which allows users to analyze Rikai datasets using own models with SQL
(**"Bring your own model"**)

Visualization
^^^^^^^^^^^^^

Carefully crafted data-visualization embedded with semantic types, especially in Jupyter notebooks,
to help you visualize and inspect your AI data without having to remember complicated raw image manipulations.

For more details, please read :doc:`quickstart`.

.. toctree::
   :maxdepth: 1

   quickstart
   sqlml
   types
   functions
   numpy
   spark
   release
   API References <./api/modules>


.. _Spark : https://spark.apache.org/
.. _Pytorch : https://pytorch.org/
.. _Tensorflow : https://www.tensorflow.org/
.. _Presto : https://prestodb.io/
.. _Jupyter : https://jupyter.org/
.. _Apache Parquet: https://parquet.apache.org/

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

