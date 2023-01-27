# Package design

## About Evaluation

About coordinate system, see [docs/perception/design.md](../perception/design.md)

### `<class> SensingEvaluationManger(...)`

- A class to execute sensing evaluation.
- This is constructed by `SensingEvaluationConfig`

#### `<class> SensingEvaluationConfig(...)`

- [perception_eval/config/sensing_evaluation_config.py](../../../perception_eval/perception_eval/config/sensing_evaluation_config.py)

  | Arguments                |       type       | Description                                                                |
  | :----------------------- | :--------------: | :------------------------------------------------------------------------- |
  | `dataset_path`           |   `List[str]`    | Dataset path(TBD supporting multiple dataset paths) .                      |
  | `frame_id`               |      `str`       | Name of coordinate system which objects respect to，`map` or `base_link` . |
  | `merge_similar_labels`   |      `bool`      | Whether merge similar labels[Reference](../perception/label.md) .          |
  | `result_root_directory`  |      `str`       | Directory path to save result of log and visualization .                   |
  | `evaluation_config_dict` | `Dict[str, Any]` | Parameters of evaluation .                                                 |
  | `load_raw_data`          |      `bool`      | Whether load pointcloud/image data from dataset .                          |

#### `evaluation_config_dict`

- In `evaluation_config_dict (Dict[str, Any])`, set sensing to `evaluation_task` and specify the other parameters.

```python
evaluation_config_dict: [Dict[str, Any]] = {
  "evaluation_task": "sensing",
  ...
}
```

- Parameters to be specified are following.

  | Arguments              |         type          | Mandatory | Description                                                        |
  | :--------------------- | :-------------------: | :-------: | :----------------------------------------------------------------- |
  | `target_uuids`         | `Optional[List[str]]` |    No     | Target objects' uuid. (Default=None)                               |
  | `box_scale_0m`         |        `float`        |    No     | Scale factor of bounding box at 0m. (Default=1.0)                  |
  | `box_scale_100m`       |        `float`        |    No     | Scale factor of bounding box at 100m ahead from ego. (Default=1.0) |
  | `min_points_threshold` |         `int`         |    No     | Number of points should be included in bounding box. (Default=1)   |

#### Execute evaluation

- Evaluate in every frame : `SensingEvaluationManger()::add_frame_result()`
