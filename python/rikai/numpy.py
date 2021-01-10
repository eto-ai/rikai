#  Copyright 2020 Rikai Authors
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

"""Wrappers and help functions to use numpy arrays transparently
in Spark / Parquet / Pytorch
"""
from __future__ import absolute_import


# Third Party
import numpy as np

# Rikai
from rikai.spark.types import NDArrayType


__all__ = ["wrap", "array", "empty"]


class ndarray(np.ndarray):  # pylint: disable=invalid-name
    """This class extends numpy ndarray, with the capability to be serialized in Spark"""

    __UDT__ = NDArrayType()


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
    """Create an numpy array

    See Also
    --------

    :py:func:`numpy.array`
    """
    return np.array(obj, *args, **kwargs).view(ndarray)


def empty(shape, dtype=float, order="C") -> np.ndarray:
    """Return a new array of giving shape and type, without initializing entries.

    The returned array can be directly used in a Spark :py:class:`~pyspark.sql.DataFrame`.

    See Also
    --------

    :py:func:`numpy.empty`
    """
    return wrap(np.empty(shape, dtype=dtype, order=order))
