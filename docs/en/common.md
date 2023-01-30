# Common items in Perception and Sensing

## [`<class> DynamicObject(...)`](../../perception_eval/perception_eval/common/object.py)

- Evaluation task: `DETECTION`, `TRACKING`, `PREDICTION`, `SENSING`

| Argument                 |                type                | Description                                                            |
| :----------------------- | :--------------------------------: | :--------------------------------------------------------------------- |
| `unix_time`              |               `int`                | Unix time .                                                            |
| `frame_id`               |             `FrameID`              | FrameID instance, where 3D objects are with respect, BASE_LINK or MAP. |
| `position`               |    `Tuple[float, float, float]`    | Position (x, y, z).                                                    |
| `orientation`            |            `Quaternion`            | Quaternion (w, x, y, z).                                               |
| `shape`                  |              `Shape`               | Shape instance that contains box size information, (width, length, depth). |
| `velocity`               |    `Tuple[float, float, float]`    | Velocity (vx, vy, vz).                                                 |
| `semantic_score`         |              `float`               | Object's confidence [0, 1].                                            |
| `semantic_label`         |          `AutowareLabel`           | Label name.                                                            |
| `pointcloud`             |          `numpy.ndarray`           | Pointcloud array.                                                      |
| `uuid`                   |               `str`                | Object's unique uuid.                                                  |
| `tracked_positions`      | `List[Tuple[float, float, float]]` | List of tracked positions.                                             |
| `tracked_orientations`   |            `Quaternion`            | List of tracked sizes.                                                 |
| `tracked_sizes`          | `List[Tuple[float, float, float]]` | List of tracked sizes.                                                 |
| `tracked_twists`         | `List[Tuple[float, float, float]]` | List of tracked velocities.                                            |
| `predicted_positions`    | `List[Tuple[float, float, float]]` | List of predicted positions.                                           |
| `predicted_orientations` |            `Quaternion`            | List of predicted orientations.                                        |
| `predicted_sizes`        | `List[Tuple[float, float, float]]` | List of predicted sizes.                                               |
| `predicted_twists`       | `List[Tuple[float, float, float]]` | List of predicted velocities.                                          |
| `predicted_confidence`   |              `float`               | List of predicted confidence.                                          |
| `visibility`             |       `Optional[Visibility]`       | Visibility status.                                                     |

## [`<class> DynamicObject2D(...)`](../../perception_eval/perception_eval/common/object2d.py)

- Evaluation task: `DETECTION2D`, `TRACING2D`, `CLASSIFICATION2D`

| Argument         |                 type                  | Description                                                     |
| :--------------- | :-----------------------------------: | :-------------------------------------------------------------- |
| `unix_time`      |                 `int`                 | Unix time .                                                     |
| `frame_id`       |               `FrameID`               | FrameID instance, where 2D objects are with respect, CAM\_\*\*. |
| `semantic_score` |                `float`                | Object's confidence [0, 1].                                     |
| `semantic_label` |              `LabelType`              | Label name.                                                     |
| `roi`            | `Optional[Tuple[int, int, int, int]]` | (x_min, y_min, width, height) of ROI. (Default: None)           |
| `uuid`           |            `Optional[str]`            | Object's UUID. (Default: None)                                  |
| `visibility`     |        `Optional[Visibility]`         | Visibility status. (Default: None)                              |

## Ground truth

### [`<class> FrameGroundTruth(...)`](../../perception_eval/perception_eval/common/dataset.py)

| Argument     |           type            | Description                                                                                             |
| :----------- | :-----------------------: | :------------------------------------------------------------------------------------------------------ |
| `unix_time`  |           `int`           | Unix time.                                                                                              |
| `frame_name` |           `str`           | Name of frame.                                                                                          |
| `objects`    |    `List[ObjectType]`     | List of ground truth objects.                                                                           |
| `ego2map`    | `Optional[numpy.ndarray]` | 4x4 matrix to transform objects with respect to base_link coordinate system map one. (Defaults to None) |
| `raw_data`   | `Optional[numpy.ndarray]` | Array of pointcloud/image. (Defaults to None)                                                           |

### [`<func> load_all_datasets(...) -> List[FrameGroundTruth]`](../../perception_eval/perception_eval/common/dataset.py)

| Argument          |                type                 | Description                                       |
| :---------------- | :---------------------------------: | :------------------------------------------------ |
| `dataset_paths`   |             `List[str]`             | List of dataset path(s).                          |
| `evaluation_task` |          `EvaluationTask`           | Name of evaluation task.                          |
| `label_converter` |          `LabelConverter`           | LabelConverter instance.                          |
| `frame_id`        | `Union[FrameID, Sequence[FrameID]]` | FrameID instance, where objects are with respect. |
| `load_raw_data`   |               `bool`                | Whether load pointcloud/image. (Default: False)   |
