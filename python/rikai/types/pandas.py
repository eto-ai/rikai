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
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_dataframe_accessor,
    register_extension_dtype,
    register_series_accessor,
)

import rikai.types as T

__all__ = ["ImageDtype", "ImageArray"]


class RikaiExtensionDtype(ExtensionDtype, ABC):
    @abstractmethod
    def from_storage(self, storage_value):
        pass


@register_extension_dtype
class ImageDtype(RikaiExtensionDtype):

    type = T.Image
    name = T.Image.__name__.lower()
    na_value = None

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError(
                "'construct_from_string' expects a string, got {}".format(
                    type(string)
                )
            )
        elif string == cls.name:
            return cls()
        else:
            raise TypeError(
                "Cannot construct a '{}' from '{}'".format(
                    cls.__name__, string
                )
            )

    @classmethod
    def construct_array_type(cls):
        return ImageArray

    def __from_arrow__(
        self, array: Union[pa.Array, pa.ChunkedArray]
    ) -> ExtensionArray:
        return ImageArray.from_dict(array.to_numpy())

    def __repr__(self):
        return "dtype('image')"

    def from_storage(self, storage_value):
        value = storage_value.get("uri", storage_value.get("data", None))
        if value is None:
            raise ValueError("Empty image")
        return T.Image(value)


@register_extension_dtype
class Box2dDtype(RikaiExtensionDtype):

    type = T.Box2d
    name = T.Box2d.__name__.lower()
    na_value = None

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError(
                "'construct_from_string' expects a string, got {}".format(
                    type(string)
                )
            )
        elif string == cls.name:
            return cls()
        else:
            raise TypeError(
                "Cannot construct a '{}' from '{}'".format(
                    cls.__name__, string
                )
            )

    @classmethod
    def construct_array_type(cls):
        return Box2dArray

    def __from_arrow__(
        self, array: Union[pa.Array, pa.ChunkedArray]
    ) -> ExtensionArray:
        return Box2dArray.from_dict(array.to_numpy())

    def __repr__(self):
        return "dtype('box2d')"

    def from_storage(self, storage_value):
        return T.Box2d(**storage_value)


# TODO I think we may want two Image subtypes for inlined vs external
#      so we don't need to deal with unpacking/packing structs etc
#      and performance can also be much better
class RikaiExtensionArray(ExtensionArray):
    def __init__(self, data):
        if isinstance(data, self.__class__):
            data = data.data
        self.data = np.array(data)

    @property
    @abstractmethod
    def dtype(self):
        pass

    def __len__(self):
        return self.shape[0]

    @property
    def size(self):
        return self.data.size

    @property
    def shape(self):
        return (self.size,)

    @property
    def ndim(self):
        return len(self.shape)

    def take(self, indices, allow_fill=False, fill_value=None):
        from pandas.api.extensions import take

        return self.__class__(
            take(
                self.data,
                indices,
                allow_fill=allow_fill,
                fill_value=fill_value,
            )
        )

    def copy(self):
        return self.__class__(self.data.copy())

    def isna(self):
        return np.array([g is None for g in self.data], dtype="bool")

    @property
    def nbytes(self):
        return self.data.nbytes

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(
        self, key: Union[int, slice, np.ndarray], value: Any
    ) -> None:
        self.data[key] = value

    def __eq__(self, other):
        return self._binop(other, operator.eq)

    def _binop(self, other, op):
        def convert_values(param):
            if not isinstance(param, self.dtype.type) and (
                isinstance(param, ExtensionArray)
                or pd.api.types.is_list_like(param)
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
        return self.__class__(data)


class ImageArray(RikaiExtensionArray):
    _dtype = ImageDtype()

    def __init__(self, data):
        super().__init__(data)

    @property
    def dtype(self):
        return self._dtype

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy=False):
        # TODO eager boxing might be slow
        return cls(np.apply_along_axis(to_img, 0, np.array(strings)))

    @classmethod
    def from_dict(cls, arr):
        return cls(np.apply_along_axis(dict_to_img, 0, arr))

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        if isinstance(scalars, np.ndarray):
            # support converting from PIL / ndarray as well
            if scalars.dtype.kind == "U":
                return cls._from_sequence_of_strings(
                    scalars, dtype=dtype, copy=copy
                )
            if copy:
                scalars = scalars.copy()
        elif isinstance(scalars, list):
            scalars = cls._from_sequence(
                np.array(scalars), dtype=dtype, copy=False
            )
        return cls(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        pass

    def __arrow_array__(self, type=None):
        # convert the underlying array values to a pyarrow Array
        from rikai.types.arrow import ImageArrowType

        dtype = ImageArrowType()
        storage = pa.array(self.data, type=dtype.storage_type)
        return pa.ExtensionArray.from_storage(dtype, storage)


to_img = np.vectorize(T.Image)
dict_to_img = np.vectorize(lambda d: T.Image(d.get("uri", d.get("data"))))


class Box2dArray(RikaiExtensionArray):
    _dtype = Box2dDtype()

    def __init__(self, data):
        super().__init__(data)

    @property
    def dtype(self):
        return self._dtype

    @classmethod
    def from_dict(cls, arr):
        return cls(np.apply_along_axis(dict_to_box, 0, arr))

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        if isinstance(scalars, np.ndarray):
            # support converting from PIL / ndarray as well
            if scalars.dtype.kind == "U":
                return cls._from_sequence_of_strings(
                    scalars, dtype=dtype, copy=copy
                )
            if copy:
                scalars = scalars.copy()
        elif isinstance(scalars, list):
            scalars = cls._from_sequence(
                np.array(scalars), dtype=dtype, copy=False
            )
        return cls(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        pass

    def __arrow_array__(self, type=None):
        # convert the underlying array values to a pyarrow Array
        from rikai.types.arrow import image_arrow_type

        dtype = image_arrow_type
        storage = pa.array(self.data, type=dtype.storage_type)
        return pa.ExtensionArray.from_storage(dtype, storage)


dict_to_box = np.vectorize(lambda d: T.Box2d(**d))


@register_dataframe_accessor("rikai")
class RikaiDataFrameAccessor:
    """support custom functionality via `df.rikai.to_table()` or `df.rikai.save()`"""

    def __init__(self, pandas_obj):
        # TODO validation
        self._obj = pandas_obj

    def to_table(self) -> pa.Table:
        """Convert DataFrame to pyarrow Table and convert nested extension
        types automatically
        """
        table = pa.Table.from_pandas(self._obj)
        converted = {}
        for name in table.column_names:
            i = table.schema.get_field_index(name)
            arr = table.column(i)
            # If we find a list of struct column
            # then convert it if it has nested extension types
            # TODO handle deeper levels of nesting
            if _is_list_of_struct(arr):
                converted_arr = self._maybe_conv_list_of_struct(name, arr)
                if converted_arr:
                    converted[name] = converted_arr
        # replace converted columns
        if len(converted) > 0:
            table = _sub(table, converted)
        return table

    def save(self, path):
        """Save this DataFrame to a parquet dataset with Rikai metadata

        Parameters
        ----------
        path: str or Path-like
            Where to save this DataFrame to
        """
        table = self.to_table()
        pq.write_to_dataset(table, str(path))

    def _maybe_conv_list_of_struct(
        self, name: str, arr: Union[pa.ListArray, pa.ChunkedArray]
    ) -> Optional[Union[pa.ListArray, pa.ChunkedArray]]:
        """Convert a list of struct column if needed. Return None
        if no conversion was performed
        """
        if isinstance(arr, pa.ChunkedArray):
            offset = 0
            conv_chunks = []
            # If it's chunked then convert each chunk
            for chunk in arr.iterchunks():
                # find the corresponding Series section
                subser = self._obj[name][offset : len(chunk)]
                was_converted, c = _maybe_convert_ext(subser, chunk)
                if was_converted:
                    conv_chunks.append(c)
                offset += len(subser)
            # TODO for now assume all chunks are converted or none are
            if len(conv_chunks) == arr.num_chunks:
                # Reassemble converted chunks
                return pa.chunked_array(conv_chunks)
        else:
            was_converted, c = _maybe_convert_ext(self._obj[name], arr)
            if was_converted:
                return c

    @classmethod
    def read_table(cls, path):
        """Read the parquet dataset under `path` into a pyarrow Table

        Parameters
        ----------
        path: str or Path-like
            Path to the root of the stored dataset
        """
        table = ds.dataset(path).to_table()
        return table

    @classmethod
    def load(cls, path):
        """Load the Rikai dataset stored at `path` into a pandas DataFrame

        Parameters
        ----------
        path: str or Path-like
            Path to the root of the stored dataset
        """
        table = cls.read_table(path)
        df = table.to_pandas()
        for name in table.column_names:
            i = table.schema.get_field_index(name)
            arr = table.column(i)
            # If we find a list of struct column, maybe convert it
            extensions = _get_nested_ext_types(arr)
            df[name] = _box_nested_extensions(df[name], extensions)
        return df


def _sub(table: pa.Table, substitutes: dict) -> pa.Table:
    """Replace table columns with given `substitutes`
    and return a new table
    """
    for name, arr in substitutes.items():
        idx = table.schema.get_field_index(name)
        table = table.set_column(idx, name, arr)
    return table


# input must be a list of struct
def _maybe_convert_ext(
    ser: pd.Series, arr: pa.ListArray
) -> (bool, pa.ListArray):
    values_arr = arr.values
    has_rikai = False
    converted = {}
    for lst in ser.values:
        for obj in lst:
            for k, v in obj.items():
                idx = values_arr.type.get_field_index(k)
                subarr = values_arr.field(idx)
                if v is not None:
                    rikai_type = _get_rikai_type(v)
                    if rikai_type:
                        has_rikai = True
                        converted[k] = _conv_arrow_array(subarr, rikai_type)
                    else:
                        converted[k] = subarr
            if len(converted) == values_arr.type.num_fields:
                break
        if len(converted) == values_arr.type.num_fields:
            break
    if has_rikai:
        values_conv = pa.StructArray.from_arrays(
            converted.values(), converted.keys()
        )
        list_conv = pa.ListArray.from_arrays(arr.offsets, values_conv)
        return True, list_conv
    return False, arr


def _get_rikai_type(v):
    # Rikai python types should have __ARROW__ property
    # OR build a reverse index from extension type registry
    if hasattr(v, "__ARROW__"):
        return v.__ARROW__


def _conv_arrow_array(arr, ext_type):
    # TODO order of fields matter in struct types
    return pa.ExtensionArray.from_storage(ext_type, arr)


def _is_list_of_struct(arr: pa.Array):
    # Is this a list-of-struct array
    return isinstance(arr.type, pa.ListType) and isinstance(
        arr.type.value_type, pa.StructType
    )


def _get_nested_ext_types(arr: pa.Array):
    """Get Rikai extension type information from nested arrays.
    If this is not a list-of-struct or it doesn't have Rikai extensions
    then don't return anything
    """
    from rikai.types.arrow import RikaiExtensionType

    if _is_list_of_struct(arr):
        return {
            field.name: field.type.to_pandas_dtype()
            for field in arr.type.value_type
            if isinstance(field.type, RikaiExtensionType)
        }
    return {}


# TODO SLOWWWWWWWWWWWWWWWWWWWWWWWWWWw
def _box_nested_extensions(
    ser: pd.Series, extensions: dict[str, RikaiExtensionDtype]
) -> pd.Series:
    # We're going to iterate through everything and box extension types
    if extensions:
        for row in ser.values:
            for obj in row:
                for name, dtype in extensions.items():
                    obj[name] = dtype.from_storage(obj[name])
    return ser
