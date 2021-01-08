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

"""Feature store analyzer"""

# Standard
import logging
from typing import Any, Dict, List

# Third Party
from pyspark.sql import DataFrame

# Rikai
from .base import Report, Rule

__all__ = ["analyze"]


def analyze(
    df: DataFrame,
    enable: List[str] = None,
    disable: List[str] = None,
    config: Dict[str, Any] = None,
) -> List[Report]:
    """Analyze a feature DataFrame

    Parameters
    ----------
    df : DataFrame
        PySpark DataFrame contains features
    enable : List[str], optional
        Enabled rules. If it is None, all rules are applied.
    disable : List[str], optional
        Disabled rules. If None is given, no rule is disabled
    config : Dict[str, Any], optional
        If set, pass the parameters to rules
    """
    # pylint: disable=unused-import,import-outside-toplevel
    import rikai.analyze.vision

    config = config or {}
    enable = set(enable or [])
    disable = set(disable or [])

    available_rules = Rule.__subclasses__()
    rules = [
        avail_rule(config=config)
        for avail_rule in available_rules
        if (not enable or avail_rule.name in enable)
        and (not disable or avail_rule.name not in disable)
    ]
    logging.info("Loading rules: %s", rules)

    reports = []
    for rule in rules:
        report = rule.apply(df)
        if report is not None:
            reports.append(report)
    return reports
