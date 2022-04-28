.. toctree::
   :maxdepth: 1

Machine Learning SQL
====================

    Make your Data Warehouse as Smart as your ML models

**Rikai** extends `Spark SQL`_ to conduct queries using Machine Learning (**ML**)
models. It is extensible to any **Model Registry**, allowing easy integration
with the existing ML infrastructures.


Setup
-----

First, let us configure a ``SparkSession`` with Rikai extension.

.. code-block:: python

    spark = (
        SparkSession
        .builder
        .appName("spark-app")
        .config("spark.jars.packages", "ai.eto:rikai_2.12:0.1.8")
        .config(
            "spark.sql.extensions",
            "ai.eto.rikai.sql.spark.RikaiSparkSessionExtensions",
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
        [FLAVOR flavor]
        [MODEL_TYPE model_type]
        [OPTIONS (key1=value1,key2=value2,...)]
        USING "uri";

        -- Describe model
        { DESC | DESCRIBE } MODEL model_name;

        -- Show all models
        SHOW MODELS;

        -- Delete a model
        DROP MODEL model_name;


Once a ML model is created via ``CREATE MODEL``, we can use it in Spark SQL:

    .. code-block:: sql

        CERATE MODEL my_resnet
        FLAVOR pytorch
        MODEL_TYPE resnet50
        USING "s3://bucket/to/resnet.pth";

        SELECT id, ML_PREDICT(my_resnet, image) FROM imagenet;


How to Use Customized ML Models
--------------------------------

Rikai creates a ML model via a combination of **Flavor** and **Model Type**.

* A **Flavor** describes the framework upon which the model was built. For example,
  Rikai offiially supports ``Tensorflow``, ``PyTorch`` and ``Sklearn`` flavors.
* A **Model Type** encaptures the interface and schema of a concrete ML model. It
  acts as an adaptor between the raw ML model input/output Tensors and
  Rikai / Spark / Pandas.

Offically supported model types:

* **PyTorch**

  * ResNet: ``resnet{18/34/50/101/152}`` and ``resnet`` (alias to ``resnet50``).
  * EfficientNet: ``efficientnet_b{0/1/2/3/4/5/6/7}``
  * FasterRCNN: ``fasterrcnn`` (alias to ``fasterrcnn_resnet50_fpn``), ``fasterrcnn_resnet50_fpn``,
    ``fasterrcnn_mobilenet_v3_large_fpn``, ``fasterrcnn_mobilenet_v3_large_320_fpn``.
  * MaskRCNN: ``maskrcnn``.
  * RetinaNet: ``retinanet``.
  * SSD: ``ssd`` and ``ssdlite``.
  * KeypointRCNN: ``keypointrcnn``.

* **Tensorflow**

  * TBD

* **Sklearn**

  * Regression: ``linear_regression``, ``logistic_regression``, ``random_forest_regression``.
  * Classification: ``random_forest_classification``
  * Dimensionality Reduction: ``pca``.


Rikai's SQL ML engine automatically looks up the python modules for an
``(flavor, model_type)`` combination.

.. code-block:: python

    rikai.{flavor}.models.{model_type}  # Official support
    rikai.contrib.{flavor}.models.{model_type}  # Third-party integration

Users can create their new model types by inherenting :py:class:`~rikai.spark.sql.model.ModelType`.




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

            # Rikai's logger adds output_schema, pre_pocessing, and post_processing as additional
            # arguments and automatically adds the flavor / rikai model spec version
            rikai.mlflow.pytorch.log_model(
                trained_model,
                "path_to_log_artifact_to",
                model_type="resnet50",
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
         'rikai.model.type': 'resnet50',
         }
        [client.set_tag(run_id, k, v) for k, v in new_tags.items()]


TorchHub Integration
--------------------
Rikai supports creating models from the `TorchHub`_ registry. Here is the minimum SQL:

    .. code-block:: sql

        CREATE MODEL resnet50
        USING "torchhub:///pytorch/vision:v0.9.1/resnet50";


It could be expanded to the equivalent and complete SQL:

    .. code-block:: sql

        CREATE MODEL my_resnet
        FLAVOR pytorch
        MODEL_TYPE resnet50
        OPTIONS (device="cpu")
        USING "torchhub:///pytorch/vision:v0.9.1/resnet50";


Most models loaded via TorchHub should adopt the pytorch flavor. In practice, `FLAVOR pytorch`
can be omitted. Usually, the `FLAVOR` keyword for specifying customized flavor.

TorchHub URI is the only required part.
    .. code-block:: verbatim

        torchhub:///repo_owner/repo_name[:tag_name]/model_name


In this case, here is the corresponding Python snippets to load the model:

    .. code-block:: python

        model = torch.hub.load('pytorch/vision:v0.9.1', 'resnet50', pretrained=True)


And the value of `rikai.contrib.torchhub.{repo_owner}.{repo_name}.{model_name}.OUTPUT_SCHEMA` will
be adopted as the `RETURNS` schema if available.

    .. warning::
        TorchHub registry is not for production usage. It is for exploring purpose. To load a
        TorchHub model, it will first download the github repo specified by the `tag_name` and then
        download the pretrained model specified by `hubconf.py` in the downloaded repo. Please be
        aware of the possible network latency and security vulnerability. In the meantime, the
        downloaded repo will be imported. It might hijack the installed Python packages.


.. _TorchHub: https://pytorch.org/hub/
.. _Spark SQL: https://spark.apache.org/sql/
