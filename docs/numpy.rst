.. toctree::
   :maxdepth: 2

Numpy and Tensor Interoperability
=================================

Numpy ``ndarray`` and Tensors are at the core of machine learning
development.
Rikai makes it effortless to work with Numpy ``ndarray`` and tensors,
and automatically converts array to the appropriate tensor format
(i.e., ``torch.Tensor`` or ``tf.Tensor``).

Work with numpy directly
~~~~~~~~~~~~~~~~~~~~~~~~

Rikai makes it super easy to work with numpy array in Spark.
At its core, ``rikai.numpy.view`` enables transparently SerDe for numpy.

    .. code-block:: python

        import numpy as np
        import PIL
        from pyspark.sql.functions import udf

        from rikai.numpy import view
        from rikai.spark.types import NDArrayType

        @udf(returnType=NDArrayType())
        def resize_mask(arr: np.ndarray) -> np.ndarray:
            """Directly work with native numpy array"""
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

Conviently, Rikai offers pytorch and tensorflow native datasets to automatically
convert numpy array into ``torch.Tensor`` or ``tf.Tensor``.

For example, using ``rikai.torch.data.Dataset`` in ``pytorch``:

    .. code-block:: python

        from rikai.torch.data import Dataset

        dataset = Dataset("s3://bucket/path/to/dataset")
        # Compatible with the official pytorch DataLoader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            num_workers=8
        )

        model = ...
        model.eval()
        for batch in loader:
            # data has already been converted from numpy array
            # to torch.Tensor
            print(batch)
            predictions = model(batch)

        # Sample output:
        # {'mask': tensor([[[0.9037, 0.9284, 0.6832, 0.5378], ..., dtype=torch.float64),
        #  'id': tensor([997]),
        #  'image': tensor([[[  5,   7,  52,  ...,  35,  74,  16],
        #  [110,  12,  45,  ..., 101,  35,  97],
        #   ...
        #  [ 25,  62,  91,  ..., 114,  71,  27]]], dtype=torch.uint8)},

Rikai supports ``tensorflow`` too:

    .. code-block:: python

        import tensorflow as tf
        import tensorflow_hub

        from rikai.tf.data import from_rikai

        dataset = (
            from_rikai(
                "s3://bucket/to/dataset",
                output_signature=(
                    tf.TensorSpec(shape=(), dtype=tf.uint8),
                    tf.TensorSpec(shape=(None, None), dtype=tf.uint8),
                ),
            )
            .map(pre_processing)
            .batch(1)
            .prefetch(tf.data.AUTOTUNE)
        )

        model = tensorflow_hub.load("https://tfhub.dev/...")
        for id, img in dataset:
            print(id, img)
            predictions = model(img)

        # Sample output:
        # tf.Tensor(99, shape=(), dtype=uint8) tf.Tensor(
        # [[ 81  39   4 ... 111  16  80]
        # ...
        # [ 15  53 121 ...   5 115  18]], shape=(128, 128), dtype=uint8)


Semantic types are Tensor convertible
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


How to develop your tensor-convertible types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~