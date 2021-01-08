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

import json
from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, explode
from pyspark.sql.types import ArrayType, StructType

# Rikai
from rikai.analyze.base import Report, Rule
from rikai.spark.types import LabelType


class LabelDistributionReport(Report):
    """Report regarding to label distribution"""

    def __init__(self, messages: Dict):
        self.messages = messages

    @property
    def json(self):
        return json.dumps(self.messages)


class LabelDistributionRule(Rule):
    """Analyze label distribution in a dataset

    Supported analyzes:

    - Imbalanced distribution
    - Insufficient examples of certain labels
    - Too many distinct labels
    """

    # Threshold of the max number distinct labels should exist.
    MAX_NUM_LABELS_KEY = "max_num_labels"
    MAX_NUM_LABELS_DEFAULT = 500

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.max_num_labels = self.get_conf(
            self.MAX_NUM_LABELS_KEY, self.MAX_NUM_LABELS_DEFAULT
        )

    @classmethod
    def name(cls):
        return "label-distribution"

    @staticmethod
    def df_to_dist(name: str, df: DataFrame) -> Dict[str, List]:
        """Convert distribution dataframe into a dict of two lists

        In such form, it would be easier for front end (web or notebook) to plot
        the distributions.
        """
        pdf = df.select("label", "count").toPandas()
        return {
            "name": name,
            "label": [str(l) for l in pdf["label"]],
            "count": pdf["count"].to_list(),
        }

    def per_example_dist(self, df: DataFrame) -> Dict:
        """Analyze dataset for the distributions of annotations per example"""
        pass

    def apply(self, df: DataFrame) -> Optional[Report]:
        """"""
        schema = df.schema

        assert isinstance(schema, StructType)

        messages = {}
        flatten_label_map = {}  # map<field_path, DataFrame>
        # TODO: find labels in nested fields
        for field in schema.fields:  # type: StructField
            if isinstance(field.dataType, StructType):
                pass
            elif isinstance(field.dataType, ArrayType):
                # only support 1 level of list
                if isinstance(field.dataType.elementType, LabelType):
                    flatten_label_map[field.name] = df.select(
                        explode(col(field.name)).alias("label")
                    )
                elif isinstance(field.dataType.elementType, StructType):
                    for subfield in field.dataType.elementType.fields:
                        if isinstance(subfield.dataType, LabelType):
                            field_name = f"{field.name}.{subfield.name}"
                            flatten_label_map[field_name] = df.select(
                                explode(col(field_name)).alias("label")
                            )
                else:
                    pass
            elif isinstance(field.dataType, LabelType):
                flatten_label_map[field.name] = df.select(
                    col(field.name).alias("label")
                )
        print(flatten_label_map)
        messages = {
            name: self.df_to_dist(name, df.groupBy("label").count().orderBy("count"))
            for name, df in flatten_label_map.items()
        }
        if messages:
            return LabelDistributionReport(messages)
        return None
