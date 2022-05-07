#  Copyright 2022 Rikai Authors
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

"""Pandas extensions

For each Rikai type, we need the following:

1. pandas ExtensionDtype
  - implement `__from_arrow__` which returns the pandas ExtensionArray
    we implement
2. pandas custom ExtensionArray
  - implement `__arrow_array__` which returns a generic arrow ExtensionArray
    filled with the arrow ExtensionType we implement
3. arrow ExtensionType
  - handles the metadata / roundtrip from parquet
  - implements `to_pandas_dtype` which is an instance of #1

References:
    https://pandas.pydata.org/docs/development/extending.html
    https://arrow.apache.org/docs/python/extending_types.html
"""

import operator
from typing import Union, Any

import numpy as np
import pandas as pd
from pandas.api.extensions import (
    ExtensionArray, ExtensionDtype, register_extension_dtype
)
import pyarrow as pa

import rikai.types as T


__all__ = ['ImageDtype', 'ImageArray']


@register_extension_dtype
class ImageDtype(ExtensionDtype):
    type = T.Image
    name = T.Image.__name__.lower()
    na_value = None

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError(
                "'construct_from_string' expects a string, got {}".format(type(string))
            )
        elif string == cls.name:
            return cls()
        else:
            raise TypeError(
                "Cannot construct a '{}' from '{}'".format(cls.__name__, string)
            )

    @classmethod
    def construct_array_type(cls):
        return ImageArray

    def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]) -> ExtensionArray:
        return ImageArray.from_dict(array.to_numpy())

    def __repr__(self):
        return "dtype('image')"


class ImageArray(ExtensionArray):

    _dtype = ImageDtype()

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype = None, copy=False):
        return cls(np.apply_along_axis(to_img, 0, np.array(strings)))

    @classmethod
    def from_dict(cls, arr):
        return cls(np.apply_along_axis(dict_to_img, 0, arr))

    def __init__(self, data):
        if isinstance(data, self.__class__):
            data = data.data
        self.data = np.array(data)

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return self.shape[0]

    @property
    def size(self):
        return self.data.size

    @property
    def shape(self):
        return self.size,

    @property
    def ndim(self):
        return len(self.shape)

    def take(self, indices, allow_fill=False, fill_value=None):
        from pandas.api.extensions import take
        return ImageArray(take(self.data, indices, allow_fill=allow_fill, fill_value=fill_value))

    def copy(self):
        return ImageArray(self.data.copy())

    def isna(self):
        return np.array([g is None for g in self.data], dtype="bool")

    @property
    def nbytes(self):
        return self.data.nbytes

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        if isinstance(scalars, np.ndarray):
            # support converting from PIL / ndarray as well
            if scalars.dtype.kind == 'U':
                return cls._from_sequence_of_strings(
                    scalars, dtype=dtype, copy=copy)
            if copy:
                scalars = scalars.copy()
        elif isinstance(scalars, list):
            scalars = np.array(scalars)
        return cls(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        pass

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key: Union[int, slice, np.ndarray], value: Any) -> None:
        self.data[key] = value

    def __eq__(self, other):
        return self._binop(other, operator.eq)

    def _binop(self, other, op):
        def convert_values(param):
            if not isinstance(param, self.dtype.type) and (
                    isinstance(param, ExtensionArray) or pd.api.types.is_list_like(param)
            ):
                ovalues = param
            else:  # Assume its an object
                ovalues = [param] * len(self)
            return ovalues

        if isinstance(other, (pd.Series, pd.Index, pd.DataFrame)):
            # rely on pandas to unbox and dispatch to us
            return NotImplemented

        lvalues = self
        rvalues = convert_values(other)

        if len(lvalues) != len(rvalues):
            raise ValueError("Lengths must match to compare")

        # If the operator is not defined for the underlying objects,
        # a TypeError should be raised
        res = [op(a, b) for (a, b) in zip(lvalues, rvalues)]

        res = np.asarray(res, dtype=bool)
        return res

    @classmethod
    def _concat_same_type(self, to_concat):
        data = np.concatenate([ga.data for ga in to_concat])
        return ImageArray(data)

    def __arrow_array__(self, type=None):
        # convert the underlying array values to a pyarrow Array
        from rikai.types.arrow import ImageArrowType
        dtype = ImageArrowType()
        storage = pa.array(self.data, type=dtype.storage_type)
        return pa.ExtensionArray.from_storage(dtype, storage)


to_img = np.vectorize(T.Image)
dict_to_img = np.vectorize(lambda d: T.Image(d.get('uri', d.get('data'))))
