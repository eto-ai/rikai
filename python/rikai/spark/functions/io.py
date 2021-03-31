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

"""I/O related PySpark UDFs."""

# Third-party
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# Rikai
from rikai.io import copy as _copy

__all__ = ["copy"]


@udf(returnType=StringType())
def copy(source: str, dest: str) -> str:
    """Copy a file from source to dest

    Parameters
    ----------
    source : str
        The source URI to copy from
    dest : str
        The destination uri or the destionation directory. If ``dest`` is
        a URI ends with a "/", it represents a directory.

    Return
    ------
    str
        Return the URI of destination.
    """
    return _copy(source, dest)
