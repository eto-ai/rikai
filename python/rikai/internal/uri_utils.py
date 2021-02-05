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

import os
from pathlib import Path
from typing import Union
from urllib.parse import urlparse


def uri_equal(uri1: str, uri2: str) -> bool:
    """Return True if two URIs are equal."""
    if uri1 == uri2:
        return True
    parsed1 = urlparse(uri1)
    parsed2 = urlparse(uri2)
    if parsed1.scheme in ["", "file"] and parsed2.scheme in ["", "file"]:
        return (
            parsed1.netloc == parsed2.netloc and parsed1.path == parsed2.path
        )
    return False


def normalize_uri(uri: Union[str, Path]) -> str:
    """Normalize URI

    Convert a file path with "file://" schema.
    Convert an relative path to absolute path

    Parameters
    ----------
    uri : str or Path

    Return
    ------
    str
        Normalized URI with schema

    """
    if isinstance(uri, Path):
        uri = str(uri.absolute())
    parsed = urlparse(uri)
    if parsed.scheme == "":
        return "file://" + os.path.abspath(uri)
    return uri
