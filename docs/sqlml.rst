.. toctree::
   :maxdepth: 1

ML-enabled SQL
==============

``Rikai`` extends Spark SQL to offer Machine Learning(**ML**)-enabled analytics.

    Make your Data Warehouse as Smart as your ML models

Rikai SQL ML is extensible to any Model Registry, no matter it is ``on-prem`` or ``cloud-native``,
``open-source`` or ``proprietary``.

As a result, ``Rikai SQL-ML`` can be easily integrated into existing machine learning infrastructure,
and allow your Data Warehouse to be as smart as your ML models.

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
        .config("spark.jars.packages", "ai.eto:rikai_2.12:0.0.21")
        .config(
            "spark.sql.extensions",
            "ai.eto.rikai.sql.spark.RikaiSparkSessionExtensions",
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

How to Use SQL ML
-----------------

Rikai extends Spark SQL with four more SQL statements:

    .. code-block:: sql

        -- Create model
        CREATE [OR REPLACE] MODEL model_name
        [OPTIONS key1=value1,key2=value2,...]
        USING "uri";

        -- Describe model
        { DESC | DESCRIBE } MODEL model_name;

        -- Show all models
        SHOW MODELS;

        -- Delete a model
        DROP MODEL model_name;


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


TorchHub Integration
--------------------
Rikai supports creating models from the `TorchHub`_ registry. Here is the minimum SQL:

    .. code-block:: sql

        CREATE MODEL resnet50
        USING "torchhub:///pytorch/vision:v0.9.1/resnet50";


It could be expanded to the equivalent and complete SQL:

    .. code-block:: sql

        CREATE MODEL resnet50
        FLAVOR pytorch
        OPTIONS (device="cpu")
        PREPROCESSOR 'rikai.contrib.torchhub.pytorch.vision.resnet50.pre_processing'
        POSTPROCESSOR 'rikai.contrib.torchhub.pytorch.vision.resnet50.post_processing'
        RETURNS array<float>
        USING "torchhub:///pytorch/vision:v0.9.1/resnet50";


Most models loaded via TorchHub should adopt the pytorch flavor. In practice, `FLAVOR pytorch`
can be omitted. Usually, the `FLAVOR` keyword for specifying customized flavor.

TorchHub URI is the only required part.
    .. code-block:: verbatim

        torchhub:///repo_owner/repo_name[:tag_name]/model_name


In this case, here is the corresponding Python snippets to load the model:

    .. code-block:: python

        model = torch.hub.load('pytorch/vision:v0.9.1', 'resnet50', pretrained=True)


Given the `repo_owner`, `repo_name` and `model_name`, here is how default PREPROCESSOR and
POSTPROCESSOR are generated:

    .. code-block:: SQL

        PREPROCESSOR 'rikai.contrib.torchhub.{repo_owner}.{repo_name}.{model_name}.pre_processing'
        POSTPROCESSOR 'rikai.contrib.torchhub.{repo_owner}.{repo_name}.{model_name}.post_processing'


And the value of `rikai.contrib.torchhub.{repo_owner}.{repo_name}.{model_name}.OUTPUT_SCHEMA` will
be adopted as the `RETURNS` schema if available.

    .. warning::
        TorchHub registry is not for production usage. It is for exploring purpose. To load a
        TorchHub model, it will first download the github repo specified by the `tag_name` and then
        download the pretrained model specified by `hubconf.py` in the downloaded repo. Please be
        aware of the possible network latency and security vulnerability. In the meantime, the
        downloaded repo will be imported. It might hijack the installed Python packages.


MLflow Integration
------------------

Rikai supports creating models from a MLflow model registry as long as a few custom tags are set
when the model is logged. To make this easier, Rikai comes with a custom model logger to add
attributes required by Rikai. This requires a simple change from your usual mlflow model logging
workflow:

    .. code-block:: python

        import rikai
        import mlflow

        with mlflow.start_run():
            # Training loop that results in a model instance

            schema = "struct<boxes:array<array<float>>, scores:array<float>, labels:array<int>>"
            pre = "rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.pre_processing"
            post = "rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.post_processing"

            # Rikai's logger adds output_schema, pre_pocessing, and post_processing as additional
            # arguments and automatically adds the flavor / rikai model spec version
            rikai.mlflow.pytorch.log_model(
                trained_model,
                "path_to_log_artifact_to",
                schema,
                pre,
                post,
                registered_model_name="my_resnet_model")

Once models are trained, you can add the model to the Rikai model catalog and query it via SparkSQL:

    .. code-block:: sql

        CREATE MODEL model_foo USING 'mlflow:///my_resnet_model';

        SELECT id, ML_PREDICT(model_foo, image) FROM df;

There are several options to refer to a mlflow model. If you specify only the mlflow model name as
in the above example (i.e., my_resnet_model), Rikai will automatically use the latest version. If
you have models in different stages as supported by mlflow, you can specify the stage like
`mlflow://my_resnet_model/production` and Rikai will select the latest version whose
`current_stage` is "production". Rikai also supports referencing a particular version number like
`mlflow://my_resnet_model/1` (Rikai distinguishes the stage and version by checking whether the
path component is a number).

If you have existing models already in mlflow model registry that didn't use Rikai's custom logger,
you can always specify flavor, schema, and pre/post-processing classes as run tags. For example:

    .. code-block:: python

        from mlflow.tracking import MlflowClient

        client = MlflowClient(<tracking_uri>)
        run_id = client.get_latest_versions("my_resnet_model", stages=['none'])
        new_tags = {
         'rikai.model.flavor': 'pytorch',
         'rikai.output.schema': 'struct<boxes:array<array<float>>, scores:array<float>, labels:array<int>>',
         'rikai.transforms.pre': 'rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.pre_processing',
         'rikai.transforms.post': 'rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.post_processing'
         }
        [client.set_tag(run_id, k, v) for k, v in new_tags.items()]

    .. warning::

        The Rikai model spec and SQL-ML API are still under heavy development so expect breaking changes!



.. _TorchHub: https://pytorch.org/hub/