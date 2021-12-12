# Yolov5 support for Rikai
`rikai-yolov5` integrates Yolov5 implemented in PyTorch with Rikai. It is based
on the [packaged ultralytics/yolov5](https://github.com/fcakyon/yolov5-pip).

## Usage
There are two ways to use `rikai-yolov5`.

Set `customized_flavor` to `yolov5` when logging the model, rikai will use
`rikai.contrib.yolov5.codegen.generate_udf` instead of
`rikai.spark.sql.codegen.pytorch.generate_udf`.

``` python
rikai.mlflow.pytorch.log_model(
    model,
    "model",
    OUTPUT_SCHEMA,
    pre_processing=pre,
    post_processing=post,
    registered_model_name=registered_model_name,
    customized_flavor="yolov5",
)
```

Another way is setting the flavor in Rikai SQL:
```
CREATE MODEL mlflow_yolov5_m
FLAVOR yolov5
OPTIONS (
  device='cpu'
)
USING 'mlflow:///{registered_model_name}';
```

## Available Options

| Name | Default Value | Description |
|------|---------------|-------------|
| conf_thres | 0.25 | NMS confidence threshold |
| iou_thres  | 0.45 | NMS IoU threshold |
| max_det    | 1000 | maximum number of detections per image |
| image_size | 640  | Image width |

Here is a sample usage of the above options:

``` sql
CREATE MODEL mlflow_yolov5_m
OPTIONS (
  device='cpu',
  iou_thres=0.5
)
USING 'mlflow:///{registered_model_name}';
```
