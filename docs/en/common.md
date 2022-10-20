# Common items in Perception and Sensing

## [`<class> DynamicObject(...)`](../../perception_eval/perception_eval/common/object.py)

| Argument                 |                type                | Description                         |
| :----------------------- | :--------------------------------: | :---------------------------------- |
| `unix_time`              |               `int`                | Unix time .                         |
| `frame_id`               |             `FrameID`              | FrameID instance, BASE_LINK or MAP. |
| `position`               |    `Tuple[float, float, float]`    | Position (x, y, z).                 |
| `orientation`            |            `Quaternion`            | Quaternion (w, x, y, z).            |
| `size`                   |    `Tuple[float, float, float]`    | Size of box (width, length, depth). |
| `velocity`               |    `Tuple[float, float, float]`    | Velocity (vx, vy, vz).              |
| `semantic_score`         |              `float`               | Object's confidence [0, 1].         |
| `semantic_label`         |          `AutowareLabel`           | Label name.                         |
| `pointcloud`             |          `numpy.ndarray`           | Pointcloud array.                   |
| `uuid`                   |               `str`                | Object's unique uuid.               |
| `tracked_positions`      | `List[Tuple[float, float, float]]` | List of tracked positions.          |
| `tracked_orientations`   |            `Quaternion`            | List of tracked sizes.              |
| `tracked_sizes`          | `List[Tuple[float, float, float]]` | List of tracked sizes.              |
| `tracked_twists`         | `List[Tuple[float, float, float]]` | List of tracked velocities.         |
| `predicted_positions`    | `List[Tuple[float, float, float]]` | List of predicted positions.        |
| `predicted_orientations` |            `Quaternion`            | List of predicted orientations.     |
| `predicted_sizes`        | `List[Tuple[float, float, float]]` | List of predicted sizes.            |
| `predicted_twists`       | `List[Tuple[float, float, float]]` | List of predicted velocities.       |
| `predicted_confidence`   |              `float`               | List of predicted confidence.       |
| `visibility`             |       `Optional[Visibility]`       | Visibility status.                  |

## Ground truth

### [`<class> FrameGroundTruth(...)`](../../perception_eval/perception_eval/common/dataset.py)

| Argument     |           type            | Description                                                                                            |
| :----------- | :-----------------------: | :----------------------------------------------------------------------------------------------------- |
| `unix_time`  |           `int`           | Unix time.                                                                                             |
| `frame_name` |           `str`           | Name of frame.                                                                                         |
| `frame_id`   |           `str`           | Frame ID of coordinate system which objects are with respect to. base_link or map.                     |
| `objects`    |   `List[DynamicObject]`   | List of ground truth objects.                                                                          |
| `ego2map`    | `Optional[numpy.ndarray]` | 4x4 matrix to transform objects with respect to base_link coordinate system map one. Defaults to None. |
| `pointcloud` | `Optional[numpy.ndarray]` | Array of pointcloud. Defaults to None.                                                                 |

### [`<func> load_all_datasets(...) -> List[FrameGroundTruth]`](../../perception_eval/perception_eval/common/dataset.py)

| Argument              |       type       | Description                                                                        |
| :-------------------- | :--------------: | :--------------------------------------------------------------------------------- |
| `dataset_paths`       |   `List[str]`    | List of dataset path(s).                                                           |
| `does_use_pointcloud` |      `bool`      | Whether load pointcloud.                                                           |
| `evaluation_task`     | `EvaluationTask` | Name of evaluation task.                                                           |
| `label_converter`     | `LabelConverter` | LabelConverter instance.                                                           |
| `frame_id`            |      `str`       | Frame ID of coordinate system which objects are with respect to. base_link or map. |
