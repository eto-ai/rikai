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
import importlib
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type, Union

import genpy
from numpy import array_str
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DataType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

__all__ = ["as_json", "as_spark_schema"]


def parse_array(message_type: str) -> Optional[Tuple[str, Optional[int]]]:
    """Parse array definition.

    Parameters
    ----------
    message_type : str
        ROS Message type, in str, i.e., `std_msgs/ByteMultiArray`.

    Return
    ------
    A tuple of `[element type, array_length]` if it is an fixed-size array.
    A tuple of `[element type, None]` if it is variable-length array.
    Return `[None, None]` if it is not an array.
    """
    if not isinstance(message_type, str):
        return None
    matched = re.match(r"(.*)\[(\d+)?\]", message_type)
    if matched:
        return matched[1], matched[2]
    return None, None


def import_message_type(message_type: str) -> Type[genpy.message.Message]:
    """Use Message Type"""
    modules = message_type.split("/")
    try:
        mod = importlib.import_module(".".join(modules[:-1] + ["msg"]))
        return getattr(mod, modules[-1])
    except ModuleNotFoundError:
        logging.error(
            'Could not load ROS message "%s", '
            "please make sure the package is installed.",
            message_type,
        )
        raise


class Converter(ABC):
    """Converter translates a ROS message or class to another type"""

    @abstractmethod
    def is_supported(self, message_type: str) -> bool:
        """Returns True if the type is convertable"""

    @abstractmethod
    def convert(self, message_type: str, value: genpy.message.Message):
        pass

    @abstractmethod
    def get_value(self, value, attr_name):
        pass

    @abstractmethod
    def build_record(self, fields):
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
        "time": lambda t: datetime.datetime.fromtimestamp(t.to_time()),
        "duration": lambda dur: dur.to_sec(),
        "byte": int,
        "byte[]": bytearray,
        "uint8[]": bytearray,
    }

    def is_supported(
        self,
        message_type: str,
    ) -> bool:
        if message_type in self.CONVERT_MAP:
            return True

        arr_type, _ = parse_array(message_type)
        if arr_type:
            return self.is_supported(arr_type)

        return False

    def convert(self, message_type: str, value: genpy.message.Message):
        if message_type in self.CONVERT_MAP:
            try:
                return self.CONVERT_MAP[message_type](value)
            except TypeError as type_err:
                logging.error(
                    "Failed to convert type=%s, value=%s, converter=%s: %s",
                    message_type,
                    value,
                    self.CONVERT_MAP[message_type],
                    type_err,
                )
                raise

        array_type_and_size = parse_array(message_type)
        if array_type_and_size:
            message_type = array_type_and_size[0]
            return [self.convert(message_type, v) for v in value]

        return None

    def get_value(self, value, attr_name):
        return getattr(value, attr_name)

    def array_type(self, value):
        return [] if value is None else list(value)

    def build_record(self, fields):
        return fields


class RikaiConverter(Converter):
    pass


class SparkSchemaConverter(Converter):
    CONVERT_MAP = {
        "bool": BooleanType,
        "char": ByteType,
        "int8": ByteType,
        "int16": ShortType,
        "int32": IntegerType,
        "int64": LongType,
        # Converting the unsigned numbers might overflow,
        # Use a large data instead.
        "uint8": ShortType,
        "uint16": IntegerType,
        "uint32": LongType,
        "uint64": LongType,
        "float32": FloatType,
        "float64": DoubleType,
        "string": StringType,
        "time": TimestampType,
        "duration": LongType,
        "byte": ByteType,
        "byte[]": BinaryType,
        "uint8[]": BinaryType,
    }

    SCHEMA_CACHE = {}

    def is_supported(
        self,
        message_type: str,
    ) -> bool:
        if message_type in self.CONVERT_MAP:
            return True

        arr_type, _ = parse_array(message_type)
        if arr_type:
            return self.is_supported(arr_type)

        return False

    def convert(self, message_type: str, value: genpy.message.Message):
        if message_type in self.CONVERT_MAP:
            return self.CONVERT_MAP[message_type]()
        array_type_and_size = parse_array(message_type)
        if array_type_and_size and self.is_supported(array_type_and_size[0]):
            return ArrayType(self.convert(array_type_and_size[0], value))
        return None

    def get_value(self, value, attr_name):
        msg_type = value._slot_types[value.__slots__.index(attr_name)]
        if msg_type in self.SCHEMA_CACHE:
            return self.SCHEMA_CACHE[msg_type]
        if self.is_supported(msg_type):
            return msg_type
        if Visitor.parse_array(msg_type):
            return msg_type
        return import_message_type(msg_type)

    def build_record(self, fields):
        return StructType([StructField(k, v) for k, v in fields.items()])


class Visitor:
    """Visitor walks through a ROS Message and converts it to
    another type system (i.e., Spark or Numpy).
    """

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
            if self.converter.is_supported(field_type):
                fields[field] = self.converter.convert(field_type, value)
                continue

            arr_type, _ = parse_array(field_type)
            if arr_type:  # is a object array
                fields[field] = [self.visit(m) for m in value]
            else:
                fields[field] = self.visit(value)

        return self.converter.build_record(fields)

    def visit_type(self, message: Union[str, Type[genpy.message.Message]]):
        """Visit a ROS Message Type

        Parameters
        ----------
        message: str or Type[genpy.message.Message]
          A string of ROS message type ``sensor_msgs/Image`` or the actual
          ROS Message class "sensor_msgs.msg.Image"

        Return
        ------
        A target system type, i.e., Spark DataType

        TODO: Converge `visit_type` with `visit` in the future.
        """
        if isinstance(message, str):
            message = import_message_type(message)

        fields = {}
        for field, field_type in self.message_fields(message):
            value = self.converter.get_value(message, field)
            if self.converter.is_supported(field_type):
                fields[field] = self.converter.convert(field_type, value)
                continue

            arr_type, _ = parse_array(field_type)
            if arr_type:  # is a object array
                fields[field] = ArrayType(self.visit_type(arr_type))
            else:
                fields[field] = self.visit_type(value)

        return self.converter.build_record(fields)


visitors = {
    "json": Visitor(JsonConverter()),
    "spark_schema": Visitor(SparkSchemaConverter()),
}


def as_json(message: genpy.message.Message) -> Dict:
    return visitors["json"].visit(message)


def as_spark_schema(message: Type[genpy.message.Message]) -> DataType:
    return visitors["spark_schema"].visit_type(message)
