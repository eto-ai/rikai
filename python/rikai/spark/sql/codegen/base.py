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

from abc import ABC, abstractmethod
from typing import Dict


class Registry(ABC):
    """Base class of a Model Registry"""

    @abstractmethod
    def resolve(self, uri: str, name: str, options: Dict[str, str]):
        """Resolve a model from a model URI.

        Parameters
        ----------
        uri : str
            Model URI
        name : str, optional
            Optional model name. Can be empty or None. If provided, it
            overrides the model name got from the model URI.
        options: dict
            Additional options passed to the model.
        """
