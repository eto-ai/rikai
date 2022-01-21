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

Rikai makes it super easy to work with numpy array in Spark
At its core, ``rikai.numpy.view`` enables transparently SerDe for numpy.

    .. code-block:: python

        import numpy as np
        import PIL
        from pyspark.sql.functions import udf

        from rikai.numpy import view
        from rikai.spark.types import NDArrayType

        @udf(returnType=NDArrayType())
        def resize_mask(arr: np.ndarray) -> np.ndarray:
            """Directly work with numpy array"""
            img = PIL.Image.fromarray(arr)
            resized_img = img.resize((32, 32))
            return view(np.asarray(resized_img))

        df = spark.createDataFrame([{
            "id": 1,
            "image": Image("s3://foo/bar/1.png"),
            # Make a view of native numpy array in Spark DataFrame
            mask=view(np.random.rand(256, 256)),
        }]).withColumn("resized", resize_mask("mask"))

        df.write.format("rikai").save("s3a://bucket/path/to/dataset")







Automatically tensor conversion for Tensorflow and Pytorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Semantic types are Tensor convertable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


How to develop your own tensor-convertable types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~