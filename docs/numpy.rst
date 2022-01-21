.. toctree::
   :maxdepth: 2

Numpy and Tensor Interoperability
=================================

Numpy ``ndarray`` and Tensors are at the core of machine learning,
especially deep learning, development.
Rikai makes it effortless to work with Numpy ndarray and tensors,
and automatically converts array to the appropriate tensor format
(i.e., ``torch.Tensor`` or ``tf.Tensor``).

Work with numpy directly
~~~~~~~~~~~~~~~~~~~~~~~~

Rikai makes it super easy to work with numpy array in Spark.
At its core, ``rikai.numpy.wrap`` enables transparently SerDe for numpy.

    .. code-block:: python

        import numpy as np
        from pyspark.sql.functions import udf

        from rikai.numpy import view

        df = spark.createDataFrame([{
            "id": 1,
            "image": Image("s3://foo/bar/1.png"),
            "annotations": [
                Row(
                    label="cat",
                    # Wrap a native numpy array in Spark DataFrame
                    mask=view(np.random.rand(256, 256)),
                )
            ],
        }])

        @udf()




Automatically tensor conversion for Tensorflow and Pytorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Semantic types are Tensor convertable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


How to develop your own tensor-convertable types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~