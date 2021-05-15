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
import functools
import shutil
from io import BytesIO
from os.path import basename, join
from pathlib import Path
from typing import BinaryIO, IO, Optional, Tuple, Union
from urllib.parse import ParseResult, urlparse

# Third Party
import requests
from pyarrow import fs

# Rikai
from rikai.logging import logger

__all__ = ["copy", "open_uri"]


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


@functools.lru_cache(maxsize=1)
def _gcsfs(project="", token=None, block_size=None):
    try:
        import gcsfs
    except ImportError as e:
        raise ImportError(
            "Please make sure gcsfs is installed via `pip install rikai[gcp]`"
        ) from e
    return gcsfs.GCSFileSystem(
        project=project, token=token, block_size=block_size
    )


def open_input_stream(uri: str) -> BinaryIO:
    """Open a URI and returns the content as a File Object."""
    parsed = urlparse(uri)
    if parsed.scheme == "gs":
        return _gcsfs().open(uri)
    else:
        filesystem, path = fs.FileSystem.from_uri(uri)
        return filesystem.open_input_file(path)


def _open_output_stream(uri: str) -> BinaryIO:
    parsed = urlparse(uri)
    if parsed.scheme == "gs":
        return _gcsfs().open(uri, mode="wb")
    else:
        filesystem, path = fs.FileSystem.from_uri(uri)
        return filesystem.open_output_stream(path)


def copy(source: str, dest: str) -> str:
    """Copy a file from source to destination, and return the URI of
    the copied file.

    Parameters
    ----------
    source : str
        The source URI to copy from
    dest : str
        The destination uri or the destination directory. If ``dest`` is
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
        scheme = parsed_dest.scheme
        if scheme == "s3":
            s3fs, source_path = fs.FileSystem.from_uri(source)
            _, dest_path = fs.FileSystem.from_uri(dest)
            s3fs.copy(source_path, dest_path)
            return dest
        elif scheme == "gs":
            _gcsfs().copy(source, dest)
            return dest

    with _open_output_stream(dest) as out_stream, open_input_stream(
        source
    ) as in_stream:
        shutil.copyfileobj(in_stream, out_stream)
    return dest


def open_uri(
    uri: Union[str, Path],
    mode: str = "rb",
    http_auth: Optional[Union[requests.auth.AuthBase, Tuple[str, str]]] = None,
) -> IO:
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
    mode : str
        the file model to open an URI
    http_auth : requests.auth.AuthBase or a tuple of (user, pass), optional
        Http credentials / auth provider when downloading via http(s)
        protocols.

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
        resp = requests.get(uri, auth=http_auth)
        return BytesIO(resp.content)
    elif parsed_uri.scheme == "gs":
        return _gcsfs().open(uri, mode=mode)
    else:
        filesystem, path = fs.FileSystem.from_uri(uri)
        return filesystem.open_input_file(path)
