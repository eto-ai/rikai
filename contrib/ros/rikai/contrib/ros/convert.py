#  Copyright (c) 2021 Rikai Authors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Conversion between ROS Message and Rikai types"""

import datetime
import re
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Generator, Dict

import genpy


__all__ = ["as_json"]


def parse_array(message_type: str) -> Optional[Tuple[str, Optional[int]]]:
    """Parse array defination.

    Parameters
    ----------
    message_type : str
        ROS Message type, in str, i.e., `std_msgs/ByteMultiArray`.

    Return
    ------
    A tuple of `[element type, array_length]` if it is an fixed-size array.
    A tuple of `[element type, None]` if it is variable-length array.
    Return None if it is not an array.
    """
    if not isinstance(message_type, str):
        return None
    matched = re.match(r"(.*)\[(\d+)?\]", message_type)
    if matched:
        return matched[1], matched[2]
    return None


class Converter(ABC):
    """Converter translates a ROS message or class to another type"""

    @abstractmethod
    def is_supported(self, message_type: str) -> bool:
        """Returns True if the type is convertable"""

    @abstractmethod
    def convert(self, message_type: str, value: genpy.message.Message):
        pass


class JsonConverter(Converter):
    """Convert ROS Message to Python Dict"""

    CONVERT_MAP = {
        "bool": bool,
        "char": int,
        "int8": int,
        "int16": int,
        "int32": int,
        "int64": int,
        # Converting the unsigned numbers might overflow.
        "uint8": int,
        "uint16": int,
        "uint32": int,
        "uint64": int,
        "float32": float,
        "float64": float,
        "string": str,
        "time": lambda rt: datetime.datetime.fromtimestamp(
            rt.secs + rt.nsecs / 1e9
        ),
        "byte": int,
        "duration": int,
        "byte[]": bytearray,
        "uint8[]": bytearray,
    }

    def is_supported(
        self,
        message_type: str,
    ) -> bool:
        if message_type in self.CONVERT_MAP:
            return True

        array_type_and_size = parse_array(message_type)
        if array_type_and_size:
            return self.is_supported(array_type_and_size[0])

        return False

    def convert(self, message_type: str, value: genpy.message.Message):
        if message_type in self.CONVERT_MAP:
            return self.CONVERT_MAP[message_type](value)

        array_type_and_size = parse_array(message_type)
        if array_type_and_size:
            message_type = array_type_and_size[0]
            return [self.convert(message_type, v) for v in value]

        return None

    def array_type(self, value):
        if value is None:
            return []

        return list(value)


class RikaiConverter(Converter):
    pass


class Visitor:
    """Visitor walks through a ROS Message and converts it to another type system (i.e., Spark or Numpy)."""

    def __init__(self, converter: Converter) -> None:
        self.converter = converter

    @staticmethod
    def message_fields(
        message: genpy.message.Message,
    ):
        return zip(message.__slots__, message._slot_types)  #

    @staticmethod
    def parse_array(message_type: str) -> Optional[Tuple[str, Optional[int]]]:
        """Parse array defination.

        Parameters
        ----------
        message_type : str
            ROS Message type, in str, i.e., `std_msgs/ByteMultiArray`.

        Return
        ------
        A tuple of `[element type, array_length]` if it is an fixed-size array.
        A tuple of `[element type, None]` if it is variable-length array.
        Return None if it is not an array.
        """
        if not isinstance(message_type, str):
            return None
        matched = re.match(r"(.*)\[(\d+)?\]", message_type)
        if matched:
            return matched[1], matched[2]
        return None

    def visit(self, message: genpy.message.Message) -> Any:
        fields = {}
        for field, field_type in self.message_fields(message):
            value = getattr(message, field)
            print("Parsing: ", field, field_type, value)
            if self.converter.is_supported(field_type):
                fields[field] = self.converter.convert(field_type, value)
                continue

            array_type_and_size = parse_array(field_type)
            if array_type_and_size:  # is a object array
                fields[field] = [self.visit(m) for m in value]
            else:
                fields[field] = self.visit(value)

        return fields


visitors = {"json": Visitor(JsonConverter())}


def as_json(message: genpy.message.Message) -> Dict:
    return visitors["json"].visit(message)
