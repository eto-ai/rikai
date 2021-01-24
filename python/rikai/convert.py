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

"""Data type conversion
"""

from enum import IntEnum

import numpy as np

__all__ = ["PortableDataType"]


class PortableDataType(IntEnum):
    """Portable identifier for data types.

    Use parquet defined data type to guarantee the consistency of serialized
    data in different languages (i.e. python and scala/java)

    Note
    ----
    This is internal use only

    """

    UINT_8 = 1
    UINT_16 = 2
    UINT_32 = 3
    UINT_64 = 4
    INT_8 = 5
    INT_16 = 6
    INT_32 = 7
    INT_64 = 8
    FLOAT = 9
    DOUBLE = 10
    BOOLEAN = 11

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> "PortableDataType":
        """Get the portable datatype from numpy dtype

        Parameters
        ----------
        cls: PortableDataType
          Class reference to PortableDataType
        dtype: numpy.dtype
          Numpy data type

        Returns
        -------
        PortableDataType
        """
        dtype = np.dtype(dtype)
        try:
            return _numpy_dtype_mapping[dtype]
        except KeyError:
            raise ValueError(f"Type {dtype} is not supported")

    def to_numpy(self) -> np.dtype:
        """Get the numpy dtype from this enum"""
        reverse_mapping = {v: k for k, v in _numpy_mapping.items()}

        return reverse_mapping[self]


_numpy_mapping = {
    np.uint8: PortableDataType.UINT_8,
    np.uint16: PortableDataType.UINT_16,
    np.uint32: PortableDataType.UINT_32,
    np.uint64: PortableDataType.UINT_64,
    np.int8: PortableDataType.INT_8,
    np.int16: PortableDataType.INT_16,
    np.int32: PortableDataType.INT_32,
    np.int64: PortableDataType.INT_64,
    np.float: PortableDataType.FLOAT,
    np.float64: PortableDataType.DOUBLE,
    np.bool: PortableDataType.BOOLEAN,
}

_numpy_dtype_mapping = {np.dtype(k): v for k, v in _numpy_mapping.items()}
