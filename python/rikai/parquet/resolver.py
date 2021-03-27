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

"""Extensive Dataset Resolver."""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Union
from urllib.parse import urlparse

import pyarrow.parquet as pq
from pyarrow.fs import FileSelector, FileSystem

from rikai.internal.uri_utils import normalize_uri
from rikai.io import _gcsfs, open_input_stream
from rikai.logging import logger

__all__ = ["register", "Resolver", "BaseResolver"]


class BaseResolver(ABC):
    """Abstract base class for the concrete resolver"""

    @abstractmethod
    def resolve(self, uri: str) -> Iterable[str]:
        """Resolve the name of a feature dataset URI, and returns a list of
        parquet file URIs.

        """

    @abstractmethod
    def get_schema(self, uri: str):
        """Return the schema of the dataset, specified by URI."""


class DefaultResolver(BaseResolver):
    """DefaultResolver supports features on local filesystem or s3.

    Supported URIs

    - "/path/to/dataset"
    - "file://path/to/dataset"
    - "s3://path/to/dataset"
    """

    SPARK_PARQUET_ROW_METADATA = b"org.apache.spark.sql.parquet.row.metadata"

    def resolve(self, uri: str) -> Iterable[str]:
        """Resolve dataset via a filesystem URI.

        Parameters
        ----------
        uri : str
            The directory / base uri for a dataset.

        Returns
        -------
        Iterator[str]
            An iterator of parquet files.
        """
        uri = normalize_uri(uri)
        parsed = urlparse(uri)
        scheme = parsed.scheme

        paths = None
        if scheme == "gs":
            fs = _gcsfs()
            glob_uri = os.path.join(uri, "*.parquet")
            logger.debug("Scan GCS directory: %s", glob_uri)
            paths = fs.glob(glob_uri)
        else:
            logger.debug("Scan pyarrow supported directory: %s", uri)
            fs, base_dir = FileSystem.from_uri(uri)
            # base_dir = parsed.netloc + parsed.path
            selector = FileSelector(
                base_dir, allow_not_found=True, recursive=True
            )
            scheme = parsed.scheme if parsed.scheme else "file"
            paths = (
                finfo.path
                for finfo in fs.get_file_info(selector)
                if finfo.path.endswith(".parquet")
            )
        return (scheme + "://" + path for path in paths)

    def get_schema(self, uri: str):
        """Get the schema of the dataset.

        Parameters
        ----------
        uri : str
            The directory URI for the dataset.

        Returns
        -------
        Dict
            Json formatted schema of the dataset.
        """

        first_parquet = next(self.resolve(uri))
        logger.debug("Resolve dataset schema from %s", first_parquet)
        metadata_file = open_input_stream(str(first_parquet))
        metadata = pq.read_metadata(metadata_file)

        kv_metadata = metadata.metadata
        try:
            return json.loads(kv_metadata[self.SPARK_PARQUET_ROW_METADATA])
        except KeyError as exp:
            raise ValueError(
                f"Parquet dataset {uri} is not created via Spark"
            ) from exp


class Resolver:
    """Extensible Dataset Resolver"""

    _DEFAULT_SCHEME = "_DEFAULT"
    # Mapping from scheme to a daaset resolver
    _RESOLVERS: Dict[str, BaseResolver] = {_DEFAULT_SCHEME: DefaultResolver()}

    @classmethod
    def reset(cls):
        """Reset Resolver for testing purpose."""
        cls._RESOLVERS = {cls._DEFAULT_SCHEME: DefaultResolver()}

    @classmethod
    def register(cls, scheme: str, resolver: BaseResolver):
        """
        Register a customize dataset resolver with given scheme, providing
        integration with feature store registeration.

        Parameters
        ----------
        scheme : str
            Feature Dataset URI scheme
        resolver : BaseResolver
            Parquet file resolver

        Raises
        ------
        KeyError
            If the same scheme name has already been registered
        """
        if scheme in cls._RESOLVERS:
            raise KeyError(f"scheme f{scheme} has already been registered")
        cls._RESOLVERS[scheme] = resolver()

    @classmethod
    def resolve(cls, uri: Union[str, Path]) -> Iterable[str]:
        """Resolve the dataset URI, and returns a list of parquet files."""
        uri = str(uri)
        parsed = urlparse(uri)
        if parsed.scheme in cls._RESOLVERS:
            logger.debug("Use extended resolver for scheme: %s", parsed.scheme)
            return cls._RESOLVERS[parsed.scheme].resolve(uri)
        return cls._RESOLVERS[cls._DEFAULT_SCHEME].resolve(uri)

    @classmethod
    def get_schema(cls, uri: str):
        """Get the schema of the dataset

        Parameters
        ----------
        uri : str
            URI of the dataset
        """
        parsed = urlparse(uri)
        if parsed.scheme in cls._RESOLVERS:
            logger.debug("Use extended resolver for scheme: %s", parsed.scheme)
            return cls._RESOLVERS[parsed.scheme].get_schema(uri)
        return cls._RESOLVERS[cls._DEFAULT_SCHEME].get_schema(uri)


def register(scheme: str):
    """
    A decorator that registers a customize dataset resolver with given scheme,
    providing integration with featurestore registeration.

    Parameters
    ----------
    scheme : str
        Feature Dataset URI scheme

    Raises
    ------
    ValueError
        If the same scheme name has already been registered

    Examples
    --------

    .. code-block:: python

        # Extend Rikai with a smart feature store
        @register("smart")
        class SmartResolver(rikai.parquet.resolver.BaseResolver):
            def resolve(self, uri: str) -> Iterable[str]:
                return smart_client.get_files(uri)

        dataset = rikai.parquet.Dataset("smart://featureA/version/1")
    """

    def wrap(resolver: BaseResolver):
        Resolver.register(scheme, resolver)
        return resolver

    return wrap
