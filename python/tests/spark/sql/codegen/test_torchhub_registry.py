from pyspark.sql import SparkSession
from utils import check_ml_predict

from rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn import (
    OUTPUT_SCHEMA,
)


def test_resnet50(spark: SparkSession):
    spark.sql(
        f"""
CREATE MODEL resnet50
FLAVOR pytorch
PREPROCESSOR 'rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.pre_processing'
POSTPROCESSOR 'rikai.contrib.torch.transforms.fasterrcnn_resnet50_fpn.post_processing'
OPTIONS (min_confidence=0.3, device="gpu", batch_size=32)
RETURNS {OUTPUT_SCHEMA}
USING "torchhub:///pytorch/vision:v0.9.1/resnet50";
    """
    )
    check_ml_predict(spark, "resnet50")
