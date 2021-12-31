
Welcome to Rikai!
=================

What is Rikai
-------------

The success of a AI product is determined by high quality of ML data.
This is the number one reason ML practioners spending the most of the efforts working on data instead
of innovating on models.
Rikai alleviates ML practioners from many of these tedious data tasks so that they can
focus on model innovation and problem solving.

Rikai is a large-scale data system specifically designed for Machine Learning workflows,
specialized in Deep Learning development over various forms of unstructured data,
for example, image, video or sensor data.

Rikai strives to offer great developer experience to assist ML engineers at each
stage of application development.

1. At its core, Rikai persists unstructured machine learning data in an `Apache Parquet`_
   based format. It handles SerDe of these unstructured data transparently via a rich collection of
   semantic types.
2. Extensive set of I/O connectors, from ETL to training (e.g., `Pytorch`_ ``Dataset``), to
   bring the familiar developer experience at each stage of ML development.
3. An SQL Engine, which extends `Spark`_ SQL with ML capability,
   that analyzes Rikai data lake with your own model ("Bring Your Own Model").
4. Carefully crafted data-visualization embedded with semantic types, especially in Jupyter notebooks.

For more details, please read :doc:`quickstart`.

.. toctree::
   :maxdepth: 1

   quickstart
   types
   spark
   sqlml
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

