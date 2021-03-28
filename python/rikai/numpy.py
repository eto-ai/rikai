#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""This module makes :py:class:`numpy.ndarray` inter-operatable with rikai
from feature engineerings in `Spark <https://spark.apache.org/>`_ to be trained
in Tensorflow and Pytorch.

>>> # Feature Engineering in Spark
>>> from rikai import numpy as np
>>> df = spark.createDataFrame([Row(mask=np.array([1, 2, 3, 4]))])
>>> df.write.format("rikai").save("s3://path/to/features")

When use the rikai data in training, the serialized numpy data will be
automatically converted into the appropriate format, i.e.,
:py:class:`torch.Tensor` in Pytorch:

>>> from rikai.torch import DataLoader
>>> data_loader = DataLoader("s3://path/to/features")
>>> next(data_loader)
{"mask": tensor([1, 2, 3])}

"""

# Third Party
import numpy as np

# Rikai
from rikai.mixin import ToNumpy
from rikai.spark.types import NDArrayType

__all__ = ["wrap", "array", "empty"]


class ndarray(np.ndarray, ToNumpy):  # pylint: disable=invalid-name
    """This class extends numpy ndarray to be serialized in Spark."""

    __UDT__ = NDArrayType()

    def to_numpy(self) -> np.ndarray:
        """Convert to a pure numpy array for compatibility"""
        # TODO: optimize it for zero-copy
        return np.copy(self)


def wrap(data: np.ndarray) -> np.ndarray:
    """Wrap a numpy array to be able to work Spark and Parquet.

    Parameters
    ----------
    data : np.ndarray
        A raw numpy array

    Returns
    -------
    np.ndarray
        A Numpy array that is compatible with Spark User Defined Type.


    Example
    -------
    >>> import numpy as np
    >>> from rikai.numpy import wrap
    >>>
    >>> arr = np.array([1, 2, 3], dtype=np.int64)
    >>> df = spark.createDataFrame([Row(id=1, mask=wrap(arr))])
    >>> df.write.format("rikai").save("s3://foo/bar")
    """
    return data.view(ndarray)


def array(obj, *args, **kwargs) -> np.ndarray:
    """Create an numpy array using the same API as :py:func:`numpy.array`.

    See Also
    --------

    :py:func:`numpy.array`
    """
    return np.array(obj, *args, **kwargs).view(ndarray)


def empty(shape, dtype=float, order="C") -> np.ndarray:
    """Return an empty :py:class:`np.ndarray`.

    The returned array can be directly used in a Spark
    :py:class:`~pyspark.sql.DataFrame`.

    See Also
    --------

    :py:func:`numpy.empty`
    """
    return wrap(np.empty(shape, dtype=dtype, order=order))
