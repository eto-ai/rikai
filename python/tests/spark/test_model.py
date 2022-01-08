#  Copyright 2022 Rikai Authors
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


from typing import Any, Callable, Dict

from mlflow.tracking import MlflowClient
from pyspark.sql import Row, SparkSession

from rikai.spark.model import create_model
from rikai.types import Image


def test_create_model(spark: SparkSession, mlflow_client: MlflowClient):
    def dynamic_preproc(options):
        from torchvision.transforms import ToTensor

        return ToTensor()

    def only_scores(options: Dict[str, Any]) -> Callable:
        # Make sure we did not actually use `rikai.contrib.torch.transforms`
        def only_score_func(batch):
            return [result["scores"].cpu().tolist() for result in batch]

        return only_score_func

    create_model(
        "vanilla",
        "mlflow:/vanilla-mlflow-no-tags/1",
        "array<float>",
        flavor="pytorch",
        preprocessor=dynamic_preproc,
        postprocessor=only_scores,
    )

    spark.sql("SHOW MODELS").show()
    df = spark.createDataFrame(
        [
            # http://cocodataset.org/#explore?id=484912
            Row(
                image=Image(
                    "http://farm2.staticflickr.com/1129/"
                    "4726871278_4dd241a03a_z.jpg"
                )
            ),
            # https://cocodataset.org/#explore?id=433013
            Row(
                image=Image(
                    "http://farm4.staticflickr.com/3726/"
                    "9457732891_87c6512b62_z.jpg"
                )
            ),
        ],
    )
    df.createOrReplaceTempView("df")

    predictions = spark.sql(
        f"SELECT ML_PREDICT(vanilla, image) as predictions FROM df"
    )
    predictions.show()
