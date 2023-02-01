# autoware_perception_evaluation

**perception_eval** is a tool to evaluate perception tasks.

## Documents

[English](docs/en/README.md) | [日本語](docs/ja/README.md)

## Overview

### Evaluate Perception & Sensing task

#### 3D tasks

| Task       |     Metrics      | Sub-metrics                         |
| :--------- | :--------------: | :---------------------------------- |
| Detection  |       mAP        | AP, APH                             |
| Tracking   |      CLEAR       | MOTA, MOTP, IDswitch                |
| Prediction |       WIP        | WIP                                 |
| Sensing    | Check Pointcloud | Detection Area & Non-detection Area |

#### 2D tasks

| Task             | Metrics  | Sub-metrics                          |
| :--------------- | :------: | :----------------------------------- |
| Detection2D      |   mAP    | AP                                   |
| Tracking2D       |  CLEAR   | MOTA, MOTP, IDswitch                 |
| Classification2D | Accuracy | Accuracy, Precision, Recall, F1score |

## Using `perception_eval`

### Evaluate with ROS node

`perception_eval` are mainly used in [tier4/driving_log_replayer](https://github.com/tier4/driving_log_replayer).
If you want to evaluated your perception results using ROS node, use driving_log_replayer or see [test/perception_lsim.py](./perception_eval/test/perception_lsim.py).

### Evaluate with your ML model

This is a simple example to evaluate your 3D detection ML model.

```python
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.manager import PerceptionEvaluationManager
from perception_eval.common.object import DynamicObject
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig

# REQUIRED:
#   dataset_path: str
#   model: Your 3D ML model

evaluation_config = PerceptionEvaluationConfig(
    dataset_paths=[dataset_path],
    frame_id="base_link",
    merge_similar_labels=False,
    result_root_directory="./data/result",
    evaluation_config_dict={
        "evaluation_task": "detection"
        "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
        "max_x_position": 100,
        "max_y_position": 100,
        "center_distance_thresholds": [1.0],
        "plane_distance_thresholds": [1.0],
        "iou_2d_thresholds": [0.5],
        "iou_3d_thresholds": [0.5],
        "min_point_numbers": [0, 0, 0, 0],
    }
    load_raw_data=True,
)

# initialize Evaluation Manager
evaluator = PerceptionEvaluationManager(evaluation_config=evaluation_config)

# this is optional
critical_object_filter_config = CriticalObjectFilterConfig(
    evaluator_config=evaluator.evaluator_config,
    target_labels=["car", "bicycle", "pedestrian", "motorbike"],
    max_x_position_list=[100, 100, 100, 100],
    max_y_position_list=[100, 100, 100, 100],
)

pass_fail_config = PerceptionPassFailConfig(
    evaluator_config=evaluator.evaluator_config,
    target_labels=["car", "bicycle", "pedestrian", "motorbike"],
    matching_threshold_list=[2.0, 2.0, 2.0, 2.0],
)

# LIDAR_TOP or LIDAR_CONCAT
sensor_name = "LIDAR_TOR"

for frame in datasets:
    unix_time = frame.unix_time
    pointcloud: numpy.ndarray = frame.raw_data[sensor_name]
    outputs = model(pointcloud)
    # create a list of estimated objects with your model's outputs
    estimated_objects = [DynamicObject(unix_time=unix_time, ...) for out in outputs]
    # add frame result
    evaluator.add_frame_result(
        unix_time=unix_time,
        ground_truth_now_frame=frame,
        estimated_objects=estimated_objects,
        ros_critical_ground_truth_objects=frame.objects,
        critical_object_filter_config=critical_object_filter_config,
        frame_pass_fail_config=pass_fail_config,
    )

scene_score = evaluator.get_scene_result()
```
