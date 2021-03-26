.. toctree::
   :maxdepth: 1

ML-enabled SQL
==============

``Rikai`` extends Spark SQL to offer Machine Learning(**ML**)-enabled analytics over
machine learning datasets.

    Make your Data Warehouse as Smart as your ML models

Similar capabilities have been provided by products, like `BigQuery ML`_ and `Redshift ML`_.
Rikai SQL ML is designed to be extensible to any Model Registry, no matter it is ``on-prem`` or ``cloud-native``,
``open-source`` or ``proprietary``.

As a result, ``Rikai SQL-ML`` can be easily integrated into existing machine learning infrastructure,
and allow your Data Warehouse as smart as your ML models.

.. warning::

    Rikai SQL-ML is still under heavily development. The syntax and implementation have not been stabilized yet.

Setup
-----

Before we can use ``Rikai SQL-ML``, we need to configure SparkSession:

.. code-block:: python

    spark = (
        SparkSession
        .builder
        .appName("spark-app")
        .config("spark.jars.packages", "ai.eto:rikai_2.12:0.0.3")
        .config(
            "spark.sql.extensions",
            "ai.eto.rikai.sql.spark.RikaiSparkSessionExtensions",
        )
        .config(
            "rikai.sql.ml.registry.file.impl",
            "ai.eto.rikai.sql.model.fs.FileSystemRegistry",
        )
        .config(
            "spark.driver.extraJavaOptions",
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
        .master("local[*]")
        .getOrCreate()
    )

    # Initialize Spark SQL-ML
    from rikai.spark.sql import init
    init(spark)

How to Use SQL ML
-----------------

Rikai extends Spark SQL with four more SQL statements:

    .. code-block:: sql

        # Create model
        CREATE [OR REPLACE] MODEL model_name
        {OPTIONS key1=value1,key2=value2,...}
        USING "uri";

        # Describe model
        { DESC | DESCRIBE } MODEL model_name;

        # Show all models
        SHOW MODELS

        # Delete a model
        DROP MODEL model_name

Rikai uses URL schema to decide which Model Registry to be used to resolve a
ML Model. Once one ML model is via ``CREATE MODEL``,
it can be used in Spark SQL directly:

    .. code-block:: sql

        CERATE MODEL model_foo USING "s3://bucket/to/spec.yaml";

        SELECT id, ML_PREDICT(model_foo, image) FROM df;


A :py:class:`~rikai.spark.sql.codegen.fs.FileSystemRegistry` is implemented as the default
model registry. It supports using a YAML spec to describe a model, for example,
the content of "s3://bucket/to/spec.yaml" can be:

    .. code-block:: yaml

        version: 1.0
        name: resnet
        model:
            uri: s3://bucket/path/to/model.pt
            flavor: pytorch
        schema: struct<boxes:array<array<float>>, scores:array<float>, labels:array<int>>
        transforms:
            pre: rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.pre_processing
            post: rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.post_processing
        options:
            batch_size: 16
            resize: 640
            min_confidence: 0.3
            use_tensorrt: true
    .. warning::

        YAML-based model spec is still under heavy development.

.. _BigQuery ML: https://cloud.google.com/bigquery-ml/docs
.. _Redshift ML: https://aws.amazon.com/redshift/features/redshift-ml/