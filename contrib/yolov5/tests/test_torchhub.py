from pathlib import Path

from pyspark.sql import SparkSession

from rikai.contrib.yolov5.transforms import OUTPUT_SCHEMA


def test_yolov5(spark: SparkSession):
    work_dir = Path().absolute().parent.parent
    image_path = f"{work_dir}/python/tests/assets/test_image.jpg"
    spark.sql(
        f"""
CREATE MODEL yolov5
FLAVOR pytorch
PREPROCESSOR 'rikai.contrib.yolov5.transforms.pre_processing'
POSTPROCESSOR 'rikai.contrib.yolov5.transforms.post_processing'
OPTIONS (device="cpu", batch_size=32)
RETURNS {OUTPUT_SCHEMA}
USING "torchhub:///ultralytics/yolov5:v5.0/yolov5s";
    """
    )
    result = spark.sql(
        f"""
    select ML_PREDICT(yolov5_way_2, '{image_path}') as pred
    """
    )

    assert len(result.first().pred.boxes) > 0
