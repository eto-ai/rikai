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

# Standard
from io import BytesIO
from os.path import basename, join
from pathlib import Path
from typing import BinaryIO, Union
from urllib.parse import ParseResult, urlparse


# Third Party
import requests
from pyarrow import fs

# Rikai
from rikai.logging import logger

__all__ = ["copy", "open_uri"]

_BUFSIZE = 8 * (2 ** 20)  # 8MB


def _normalize_uri(uri: str) -> str:
    parsed = urlparse(uri)
    scheme = parsed.scheme
    if not scheme:
        scheme = "file"
    elif scheme in ["s3a", "s3n"]:
        scheme = "s3"
    return ParseResult(
        scheme=scheme,
        netloc=parsed.netloc,
        path=parsed.path,
        query=parsed.query,
        fragment=parsed.fragment,
        params=parsed.params,
    ).geturl()


def copy(source: str, dest: str) -> str:
    """Copy a file from source to destination, and return the URI of
    the copied file.

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
    source = _normalize_uri(source)
    dest = _normalize_uri(dest)
    parsed_source = urlparse(source)
    if dest and dest.endswith("/"):
        dest = join(dest, basename(parsed_source.path))
    parsed_dest = urlparse(dest)
    logger.debug("Copying %s to %s", source, dest)

    if parsed_dest.scheme == parsed_source.scheme:
        # Direct copy with the same file system
        if parsed_dest.scheme == "s3":
            s3fs, source_path = fs.FileSystem.from_uri(source)
            _, dest_path = fs.FileSystem.from_uri(dest)
            s3fs.copy(source_path, dest_path)
            return dest

    # TODO: find better i/o utilis to copy between filesystems
    filesystem, dest_path = fs.FileSystem.from_uri(dest)
    with filesystem.open_output_stream(dest_path) as out_stream:
        src_fs, src_path = fs.FileSystem.from_uri(source)
        with src_fs.open_input_stream(src_path) as in_stream:
            while True:
                buf = in_stream.read(_BUFSIZE)
                if not buf:
                    break
                out_stream.write(buf)
    return dest


def open_uri(uri: Union[str, Path], mode="rb") -> BinaryIO:
    """Open URI for read.

    It supports the following URI pattens:

    - File System: ``/path/to/file`` or ``file:///path/to/file``
    - AWS S3: ``s3://``
    - Google Cloud Storage: ``gs://``
    - Http(s): ``http://`` or ``https://``

    Parameters
    ----------
    uri : str or :py:class:`~pathlib.Path`
        URI of the object

    Return
    ------
    File
        A file-like object for sequential read.
    """
    if isinstance(uri, Path):
        return uri.open()
    parsed_uri = urlparse(uri)
    if not parsed_uri.scheme:
        # This is a local file
        return open(uri, mode=mode)
    elif parsed_uri.scheme in ("http", "https"):
        resp = requests.get(uri)
        return BytesIO(resp.content)
    else:
        filesystem, path = fs.FileSystem.from_uri(uri)
        return filesystem.open_input_file(path)
