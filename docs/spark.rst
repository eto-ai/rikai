.. toctree::
   :maxdepth: 1

URI Scheme
==========
Remember to prefix your S3 paths with `s3a` instead of `s3` or `s3n`.

Local Spark Setup
=================
If you're running Spark locally, you'll need to add the rikai jar when creating the Spark session:

  .. code-block:: python

      spark = (
         SparkSession
            .builder
            .appName('rikai-quickstart')
            .config('spark.jars.packages', 'ai.eto:rikai:0.0.1')
            .master('local[*]')
            .getOrCreate()
      )

If you want to read/write data from/to S3, you will need additional setup:

1. Setup `AWS credentials <https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html>`_
   or specify them directly as Spark config
2. Add :code:`hadoop-aws` and :code:`aws-java-sdk` jars to your Spark classpath. Make sure you download versions
   that match. For example, if you have apache spark 3.0.1 with hadoop 2.7.4 setup, then you should
   use :code:`hadoop-aws v2.7.4`. You can then see on maven that this pairs with :code:`aws-java-sdk v1.7.4`.
3. Specify additional options when creating the Spark session:

  .. code-block:: python

     spark = (
         SparkSession
         .builder
         .appName('rikai-quickstart')
         .config('spark.jars.packages', 'ai.eto:rikai:0.0.1')
         .config("spark.driver.extraJavaOptions", "-Dcom.amazonaws.services.s3.enableV4=true")
         .config("spark.executor.extraJavaOptions", "-Dcom.amazonaws.services.s3.enableV4=true")
         .master("local[*]")
         .getOrCreate()
     )

Note that for hadoop 2.7.x you may need to configure the aws endpoints. See
`hadoop-aws <https://hadoop.apache.org/docs/current/hadoop-aws/tools/hadoop-aws/index.html>`_
documentation for details.

Databricks
==========

If you are using Databricks, you shouldn't need to manually configure the Spark options and
classpath. Please follow
`Databricks documentation <https://docs.databricks.com/libraries/index.html>`_
and install both the `python package from pypi <https://pypi.org/project/rikai/>`_ and
the `jar from maven <https://mvnrepository.com/artifact/ai.eto/rikai>`_.