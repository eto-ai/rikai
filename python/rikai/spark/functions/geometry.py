#  Copyright 2021 Rikai Authors
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

"""Geometry related PySpark UDF"""

# Third Party
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# Rikai
from rikai.types.geometry import Box2d

__all__ = ["area"]


@udf(returnType=FloatType())
def area(bbox: Box2d) -> float:
    """A UDF to calculate the area of a bounding box."""
    return bbox.area
