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


"""Test Vision Related Rules"""

# Third Party
from pyspark.sql import Row

# Rikai
from rikai.testing import SparkTestCase
from rikai.vision import Label
from rikai.analyze.vision import LabelDistributionRule


class VisionRulesTest(SparkTestCase):
    """Tests for vision related rules"""

    def test_first_level_label(self):
        """Test label in the first level structure"""
        labels = ["a"] * 5 + ["b"] * 10
        df = self.spark.createDataFrame([Row(Label(i)) for i in labels], ["v"])
        report = LabelDistributionRule({}).apply(df)
        self.assertIsNotNone(report)
        messages = report.messages
        self.assertTrue("v" in messages)
        self.assertEqual(
            messages["v"], {"label": ["a", "b"], "count": [5, 10], "name": "v"}
        )

    def test_list_of_structs(self):
        """Test the labels in a list of structures in each row"""
        df = self.spark.createDataFrame(
            [
                Row(
                    id="a",
                    annotations=[
                        Row(label=Label("a"), v=1),
                        Row(label=Label("b"), v=2),
                    ],
                ),
                Row(
                    id="b",
                    annotations=[
                        Row(label=Label("b"), v=3),
                        Row(label=Label("d"), v=4),
                    ],
                ),
            ],
        )
        report = LabelDistributionRule({}).apply(df)
        messages = report.messages
        self.assertEqual(
            {
                "annotations.label": {
                    "name": "annotations.label",
                    "label": ["d", "a", "b"],
                    "count": [1, 1, 2],
                }
            },
            messages,
        )
