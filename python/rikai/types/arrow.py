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

"""Arrow extensions
"""
import pyarrow as pa

__all__ = ['ImageArrowType', 'Box2dArrowType', 'image_arrow_type', 'box2d_arrow_type']


class ImageArrowType(pa.ExtensionType):

    def __init__(self):
        pa.ExtensionType.__init__(self, pa.struct([pa.field('uri', pa.string()), pa.field('data', pa.binary())]),
                                  "rikai.image")

    def __arrow_ext_serialize__(self):
        # since we don't have a parameterized type, we don't need extra
        # metadata to be deserialized
        return b''

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        # return an instance of this subclass given the serialized
        # metadata.
        return ImageArrowType()

    def to_pandas_dtype(self):
        from rikai.types.pandas import ImageDtype
        return ImageDtype()


image_arrow_type = ImageArrowType()
pa.register_extension_type(image_arrow_type)


class Box2dArrowType(pa.ExtensionType):

    def __init__(self):
        pa.ExtensionType.__init__(self,
                                  pa.struct([
                                      pa.field('xmax', pa.float64()),
                                      pa.field('xmin', pa.float64()),
                                      pa.field('ymax', pa.float64()),
                                      pa.field('ymin', pa.float64())
                                  ]),
                                  "rikai.box2d")

    def __arrow_ext_serialize__(self):
        # since we don't have a parameterized type, we don't need extra
        # metadata to be deserialized
        return b''

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        # return an instance of this subclass given the serialized
        # metadata.
        return Box2dArrowType()

    def to_pandas_dtype(self):
        from rikai.types.pandas import Box2dDtype
        return Box2dDtype()


box2d_arrow_type = Box2dArrowType()
pa.register_extension_type(box2d_arrow_type)
