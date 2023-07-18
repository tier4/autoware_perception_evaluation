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

### Dataset format

We support `T4Dataset` format. This has same structure with [NuScenes](https://www.nuscenes.org/nuscenes).
The expected dataset directory tree is shown as below.

```shell
data_root/
    │── annotation/     ... annotation information in json format.
    │   │── sample.json
    │   │── sample_data.json
    │   │── sample_annotation.json
    │   └── ...
    └── data/           ... raw data.
        │── LIDAR_CONCAT/  # LIDAR_TOP is also OK.
        └── CAM_**/
```

## Using `perception_eval`

### Evaluate with ROS

`perception_eval` is mainly used in [tier4/driving_log_replayer](https://github.com/tier4/driving_log_replayer) that is a tool to evaluate output of [autoware](https://github.com/autowarefoundation/autoware).
If you want to evaluate your perception results through ROS, use `driving_log_replayer` or refer [test/perception_lsim.py](./perception_eval/test/perception_lsim.py).

### Evaluate with your ML model

This is a simple example to evaluate your 3D detection ML model.
Basically, most parts of the codes are same with [test/perception_lsim.py](perception_eval/test/perception_lsim.py), so please refer it.

```python
from perception_eval.config import PerceptionEvaluationConfig
from perception_eval.manager import PerceptionEvaluationManager
from perception_eval.common.object import DynamicObject

# REQUIRED:
#   dataset_path: str
#   model: Your 3D ML model

evaluation_config = PerceptionEvaluationConfig(
    dataset_paths=[dataset_path],
    frame_id="base_link",
    merge_similar_labels=False,
    result_root_directory="./data/result",
    evaluation_config_dict={"evaluation_task": "detection",...},
    load_raw_data=True,
)

# initialize Evaluation Manager
evaluator = PerceptionEvaluationManager(evaluation_config=evaluation_config)

for frame in datasets:
    unix_time = frame.unix_time
    pointcloud: numpy.ndarray = frame.raw_data["lidar"]
    outputs = model(pointcloud)
    # create a list of estimated objects with your model's outputs
    estimated_objects = [DynamicObject(unix_time=unix_time, ...) for out in outputs]
    # add frame result
    evaluator.add_frame_result(
        unix_time=unix_time,
        ground_truth_now_frame=frame,
        estimated_objects=estimated_objects,
    )

scene_score = evaluator.get_scene_result()
```
