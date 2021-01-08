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

from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from typing import Any, Dict, List, Optional, Union

from pyspark.sql import DataFrame

__all__ = ["Report", "Rule"]


class Report(ABC):  # pylint: disable=too-few-public-methods
    """Report generated from a rule

    Each report contains one or more messages from a defined rule.

    Requirements:
     - This class should be friendly to be sent via HTTP.
     - It should be display-able in a Jupyter notebook.
    """

    @abstractproperty
    def json(self) -> Dict[str, Union[List, Dict, str, int, float]]:
        """Convert Report to a JSON payload"""


class Rule(ABC):
    """The interface for analyzing a feature Dataset

    Rules will be auto registered, thus each rule must have a unique ``name``.

    Users can use a global config (dict) object to control the behavior of each individual rule.
    """

    CONFIG_KEY_PREFIX = "rikai.analyze.rules"

    def __init__(self, config: Dict[str, Any]) -> None:
        """Construct a Rule with provided config

        Parameters
        ----------
        config : Dict
            Configuration. A key-value object to contain the configurations for all the rules.
            Each rule has its own namespace for the configurations, i.e.,
            ``<CONFIG_KEY_PREFIX>.<rule-name>.<config-name>``.
        """
        self.config = config

    @abstractclassmethod
    def name(cls):
        """Unique rule name"""

    def __repr__(self) -> str:
        return f"Rule({self.name})"

    def get_conf(self, key: str, default: Any = None) -> Any:
        """Get configuration value

        Parameters
        ----------
        key : str
            The configuration key under rule's namespace.
        default : Any, optional
            Default value.
        """
        if not key.startswith(self.CONFIG_KEY_PREFIX):
            key = f"{self.CONFIG_KEY_PREFIX}.{self.name}.{key}"
        return self.config.get(key, default)

    @abstractmethod
    def apply(self, df: DataFrame) -> Optional[Report]:
        """Apply the rule to a Spark DataFrame to generate one report.

        Parameters
        ----------
        df : :py:class:`pyspark.sql.DataFrame`
            PySpark DataFrame

        Return
        ------
        Report, optional
            Return the generated report. Return None if this rule does not apply.
        """
