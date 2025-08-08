# Package design

## About software

- Autoware Evaluator
  - Evaluation system for autoware on cloud
  - This manages apps for evaluation including driving_log_replayer
- [driving_log_replayer](https://github.com/tier4/driving_log_replayer)
  - ROS package to evaluate autoware
- autoware_perception_evaluation
  - Name of repository for evaluating Perception/Sensing tasks

## About evaluation

### Information output by driving_log_replayer

### Coordinate system of autoware objects

| Autoware module                   | Coordinate system |
| :-------------------------------- | :---------------: |
| detection/objects                 |    `base_link`    |
| tracking/objects                  |       `map`       |
| objects (-> prediction/objects ?) |       `map`       |
| pointcloud                        |    `base_link`    |

### `<class> PerceptionEvaluationManager(...)`

- A class to execute perception evaluation
- This is constructed by `PerceptionEvaluationConfig`

#### `<class> PerceptionEvaluationConfig(...)`

- Fot the details, see [perception_eval/config/perception_evaluation_config.py](../../../perception_eval/perception_eval/config/perception_evaluation_config.py).

- Parameters of `PerceptionEvaluationConfig` are following.

  | Arguments                |            type             | Description                                            |
  | :----------------------- | :-------------------------: | :----------------------------------------------------- |
  | `dataset_path`           |         `List[str]`         | Dataset path(TBD supporting multiple dataset paths)    |
  | `frame_id`               | `Union[str, Sequence[str]]` | FrameID in string, where objects are with respect.     |
  | `result_root_directory`  |            `str`            | Directory path to save result of log and visualization |
  | `evaluation_config_dict` |      `Dict[str, Any]`       | Parameters of evaluation                               |
  | `load_raw_data`          |           `bool`            | Whether load pointcloud/image data from dataset        |

##### `evaluation_config_dict`

- In `evaluation_config_dict (Dict[str, Any])`，set detection/tracking/prediction to `evaluation_task` and specify the other parameters

  ```python
  evaluation_config_dict: [Dict[str, Any]] = {
    "evaluation_task": "detection"/"tracking"/"prediction" or "detection2d/tracking2d/classification2d",
    ...
  }
  ```

- Each parameter can be categorized in **1.Thresholds for filtering objects**, **2.Parameters for label settings**, **3.Thresholds to determine TP/FP/FN**

  - **1. Thresholds for filtering objects to generate `DynamicObjectWithPerceptionResult`**

    | Arguments              |     type      |     Mandatory      | Description                                                                                                     |
    | :--------------------- | :-----------: | :----------------: | :-------------------------------------------------------------------------------------------------------------- |
    | `target_labels`        |  `List[str]`  |         No         | List of name of target labels. If None, all labels will be evaluated.                                           |
    | `ignore_attributes`    |  `List[str]`  |         No         | List of original name or attribute of labels to be filtered out. If None, all labels will be evaluated.         |
    | `max_x_position`       |    `float`    |         \*         | Maximum x position of area to be evaluated (Only 3D)                                                            |
    | `max_y_position`       |    `float`    |         \*         | Maximum y position of area to be evaluated (Only 3D)                                                            |
    | `max_distance`         |    `float`    |         \*         | Maximum distance from `base_link` of ego to be evaluated (Only 3D)                                              |
    | `min_distance`         |    `float`    |         \*         | Minimum distance from `base_link` of ego to be evaluated (Only 3D)                                              |
    | `min_point_numbers`    |  `List[int]`  | Yes (in Detection) | Minimum number of pointcloud included in GT's bounding box. If `min_point_numbers=0`, evaluate all GT (Only 3D) |
    | `max_matchable_radii`  | `List[float]` |         No         | Maximum distance of matching objects to be allowed (Default: `None`)                                            |
    | `confidence_threshold` |    `float`    |         No         | Threshold of confidence of estimation                                                                           |
    | `target_uuids`         |  `List[str]`  |         No         | List of GTs' ID. Specify if you want to evaluated specific objects                                              |

    \* It is necessary to specify either **max_x/y_position** or **max/min_distance**. Another groups must be `None`.

  - **2. Parameters for label settings**

    | Arguments                |  type  | Mandatory | Description                                                               |
    | :----------------------- | :----: | :-------: | :------------------------------------------------------------------------ |
    | `label_prefix`           | `str`  |    Yes    | Type of label to be used ("autoware", "traffic_light")                    |
    | `merge_similar_labels`   | `bool` |    No     | Whether to merge similar labels (Default: `False`)                        |
    | `allow_matching_unknown` | `bool` |    No     | Whether allow to match unknown label estimation and GT (Default: `False`) |
    | `count_label_number`     | `bool` |    No     | Whether to count the number of loaded labels (Default: `True`)            |

  - **3. Thresholds to determine TP/FP/FN for `DynamicObjectWithPerceptionResult`**

    - For `classification2d`, there is no need to specify the following parameters.

    | Arguments                        |     type      | Mandatory | Description                            |
    | :------------------------------- | :-----------: | :-------: | :------------------------------------- |
    | `center_distance_thresholds`     | `List[float]` |    Yes    | Thresholds of center distance          |
    | `center_distance_bev_thresholds` | `List[float]` |    Yes    | Thresholds of center distance in BEV   |
    | `plane_distance_thresholds`      | `List[float]` |    Yes    | Thresholds of plane distance (Only 3D) |
    | `iou_2d_thresholds`              | `List[float]` |    Yes    | Thresholds of IoU in 2D                |
    | `iou_3d_thresholds`              | `List[float]` |    Yes    | Thresholds of IoU in 3D (Only 3D)      |

- **Error cases in setting parameters**

  - **1. Set the invalid `evaluation_task` except of `detection/tracking/prediction/detection2d/tracking2d/classification2d`**

  ```python
  evaluation_config_dict = {
    "evaluation_task": "foo",  # <-- Set "foo"
    "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
    "ignore_attributes": [],
    "max_x_position": 102.4,
    "max_y_position": 102.4,
    "min_point_numbers": [0, 0, 0, 0],
    "label_prefix": "autoware",
    "merge_similar_labels": False,
    "allow_matching_unknown": True,
    "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
    "center_distance_bev_thresholds": [[1.0, 1.0, 1.0, 1.0]],
    "plane_distance_thresholds": [2.0, 3.0],
    "iou_2d_thresholds": [0.5],
    "iou_3d_thresholds": [0.5],
  }
  ```

  ```shell
  # Exception
  >>  ValueError: Unsupported task: 'foo'
      Supported tasks: ['detection', 'tracking', 'prediction', 'detection2d', 'tracking2d', 'classification2d']
  ```

  - **2. Unset either `max_x/y_position` and `max/min_distance` or set both of them**

    - Set both of them

    ```python
    evaluation_config_dict = {
      "evaluation_task": "detection",
      "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
      "ignore_attributes": [],
      "max_x_position": 102.4,
      "max_y_position": 102.4,
      "max_distance": 100.0,
      "min_distance": 10.0,
      "min_point_numbers": [0, 0, 0, 0],
      "label_prefix": "autoware",
      "merge_similar_labels": False,
      "allow_matching_unknown": True,
      "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "center_distance_bev_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "plane_distance_thresholds": [2.0, 3.0],
      "iou_2d_thresholds": [0.5],
      "iou_3d_thresholds": [0.5],
    }
    ```

    ```shell
    # Exception
    >> RuntimeError: Either max x/y position or max/min distance should be specified
    ```

    - Unset either both of them

    ```python
    evaluation_config_dict = {
      "evaluation_task": "detection",
      "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
      "ignore_attributes": [],
      # "max_x_position": 102.4,  # <-- comment-out all
      # "max_y_position": 102.4,
      # "max_distance": 100.0,
      # "min_distance": 10.0,
      "min_point_numbers": [0, 0, 0, 0],
      "label_prefix": "autoware",
      "merge_similar_labels": False,
      "allow_matching_unknown": True,
      "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "center_distance_bev_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "plane_distance_thresholds": [2.0, 3.0],
      "iou_2d_thresholds": [0.5],
      "iou_3d_thresholds": [0.5],
    }
    ```

    ```shell
    # Exception
    >> RuntimeError: Either max x/y position or max/min distance should be specified
    ```

  - **3. Unset `min_point_numbers` in detection evaluation**

    ```python
    evaluation_config_dict = {
      "evaluation_task": "detection",  # <-- Set "detection"
      "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
      "ignore_attributes": [],
      "max_x_position": 102.4,
      "max_y_position": 102.4,
      # "min_point_numbers": [0, 0, 0, 0],  # <-- comment-out "min_point_numbers"
      "label_prefix": "autoware",
      "merge_similar_labels": False,
      "allow_matching_unknown": True,
      "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "center_distance_bev_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "plane_distance_thresholds": [2.0, 3.0],
      "iou_2d_thresholds": [0.5],
      "iou_3d_thresholds": [0.5],
    }
    ```

    ```shell
    # Exception
    >> RuntimeError: In detection task, min point numbers must be specified
    ```

  - **4. Specify parameters besides parameters of `MetricsConfig`**

    - About each MetricsConfig，see [perception_eval/evaluation/metrics/config](../../../perception_eval/perception_eval/evaluation/metrics/config/)

    ```python
    evaluation_config_dict = {
      "evaluation_task": "detection",
      "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
      "ignore_attributes": [],
      "max_x_position": 102.4,
      "max_y_position": 102.4,
      "label_prefix": "autoware",
      "min_point_numbers": [0, 0, 0, 0],
      "merge_similar_labels": False,
      "allow_matching_unknown": True,
      "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "center_distance_bev_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "plane_distance_thresholds": [2.0, 3.0],
      "iou_2d_thresholds": [0.5],
      "iou_3d_thresholds": [0.5],
      "min_point_numbers": [0, 0, 0, 0],
      "foo_thresholds": [0.8],  # <-- set "foo_thresholds"
    }
    ```

    ```shell
    # Exception
    >>  perception_eval.evaluation.metrics.metrics_score_config.MetricsParameterError: MetricsConfig for 'EvaluationTask.DETECTION'
        Unexpected parameters: {'foo_thresholds'}
        Usage: {'plane_distance_thresholds', 'iou_3d_thresholds', 'center_distance_thresholds', 'target_labels', 'iou_2d_thresholds'}
    ```

### `<class> CriticalObjectFilterConfig(...)`

- An interface to determine target objects dynamically
- Specify in every frame, not in initialization of `PerceptionEvaluationManager`
- See [perception_eval/evaluation/result/perception_frame_config](../../../perception_eval/perception_eval/evaluation/result/perception_frame_config.py)

| Arguments                   |             type             |    Mandatory    | Description                                                                                                     |
| :-------------------------- | :--------------------------: | :-------------: | :-------------------------------------------------------------------------------------------------------------- |
| `evaluator_config`          | `PerceptionEvaluationConfig` |       Yes       | Configuration settings which `PerceptionEvaluationManager` has                                                  |
| `target_labels`             |         `List[str]`          |       No        | List of name of target labels                                                                                   |
| `ignore_attributes`         |         `List[str]`          |       No        | List of original name or attribute of labels to be filtered out.                                                |
| `max_x_position_list`       |        `List[float]`         |       \*        | Maximum x position of area to be evaluated (Only 3D)                                                            |
| `max_y_position_list`       |        `List[float]`         |       \*        | Maximum y position of area to be evaluated (Only 3D)                                                            |
| `max_distance_list`         |        `List[float]`         |       \*        | Maximum distance from `base_link` of ego to be evaluated (Only 3D)                                              |
| `min_distance_list`         |        `List[float]`         |       \*        | Minimum distance from `base_link` of ego to be evaluated (Only 3D)                                              |
| `min_point_numbers`         |         `List[int]`          | Yes (Detection) | Minimum number of pointcloud included in GT's bounding box. If `min_point_numbers=0`, evaluate all GT (Only 3D) |
| `confidence_threshold_list` |        `List[float]`         |       No        | Threshold of confidence of estimation                                                                           |
| `target_uuids`              |         `List[str]`          |       No        | List of GTs' ID. Specify if you want to evaluated specific objects                                              |

\* It is necessary to specify either **max_x/y_position_list** or **max/min_distance_list**. Another groups must be `None`.

### `<class> PerceptionPassFailConfig(...)`

- A class to decide Pass / Fail. For Pass/Fail decision, determine TP/FP by **Plane distance**.
- Specify in every frame, not in initializing `PerceptionEvaluationManager`.
- For the details, see [perception_eval/evaluation/result/perception_frame_config](../../../perception_eval/perception_eval/evaluation/result/perception_frame_config.py).

| Arguments                 |             type             | Mandatory | Description                                                                                                            |
| :------------------------ | :--------------------------: | :-------: | :--------------------------------------------------------------------------------------------------------------------- |
| `evaluator_config`        | `PerceptionEvaluationConfig` |    Yes    | Configuration settings which `PerceptionEvaluationManager` has.                                                        |
| `target_labels`           |         `List[str]`          |    No     | List of name of target labels.                                                                                         |
| `matching_threshold_list` |        `List[float]`         |    No     | Thresholds of matching. For 3D evaluation, plane distance will be used. For 2D detection/tracking, IoU2D will be used. |

### Execute evaluation

- Evaluation in every frame : `PerceptionEvaluationManager::add_frame_result()`
  - Compute metrics score from estimations and GTs in every frame
- Evaluation for scene : `PerceptionEvaluationManager()::get_scene_result()`
  - Compute metrics score for scene (=all frame)

## Data structure

- For the details，see [data_structure.md](./data_structure.md)

## Metrics

- For the details，see [metrics.md](./metrics.md)

### Detection

| Metrics | Sub Metrics |
| ------- | ----------- |
| mAP     | AP, APH     |

### Tracking

|  Metrics   |      Sub Metrics      |
| :--------: | :-------------------: |
|   CLEAR    | MOTA, MOTP, ID switch |
| HOTA (TBD) |   HOTA, DetA, AssA    |

### Prediction (WIP)

| Metrics | Sub Metrics |
| :-----: | :---------: |

### Classification

| Metrics  |             Sub Metrics              |
| :------: | :----------------------------------: |
| Accuracy | Accuracy, Precision, Recall, F1score |
