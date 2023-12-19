# Data structure

## Result of pointcloud detection

1. ~Position of pointcloud included in bounding box(List[Tuple[float]])~
   - pros: Available to leave decision reason.
   - cons: Too much amount of computation and data.
2. **Bounding box の中の点群の個数 (int)**
   - pros: Available to leave decision reason. `.json` is easy to see.
   - cons: Too much amount of computation.
3. ~Existence of pointcloud included in bounding box (bool)~
   - pros: `result.json` is easy to see. Less amount of computation.
   - cons: Needs to visualize when we want to observe pointcloud status.

- In `perception_eval`, apply 2. Because,
  - For 1, the details of pointcloud's position can see with viewer or rosbag.

## Data structure per one Evaluation

- Catalog: A set of scenario
  - Example: turn right UseCase = (UC-0001, UC-0003, UC-0006)

```yaml
- Catalog
  - List[Scenario]
```

- Scenario: An unit of evaluation

  - There are two types of scenario, DataBase(DB) and UseCase(UC).

```yaml
- Scenario
  - List[Scene]
```

- Scene: An unit of one sequence of rosbag.
  - A set of data constructed by one rosbag.
  - A set of one rosbag and one .pcd file and some .jpeg files.

```yaml
- Scene
  - List[PerceptionFrameResult]
```

## Data structure per one Frame

### `<class> SensingFrameResult(...)`

Sensing result for pointcloud in detection/non-detection area at one frame.
For the details, see [perception_eval/result/sensing/sensing_frame_result.py](../../../perception_eval/perception_eval/result/sensing/sensing_frame_result.py)

| Argument               |         type         | Description             |
| :--------------------- | :------------------: | :---------------------- |
| `sensing_frame_config` | `SensingFrameConfig` | Configuration settings. |
| `unix_time`            |        `int`         | UNIX timestamp          |
| `frame_name`           |        `str`         | Name of frame.          |

#### Execute evaluation : `<func> SensingFrameResult.evaluate_frame(...)`

Evaluate false position pointcloud detection in detection/non-detection area.

- Evaluation in detection area
  - Check whether number of pointcloud included in GT, `pointcloud_for_detection`, is more than threshold value．
- Evaluate in non-detection area
  - Check whether there is no pointcloud in non-detection area.

| Argument                       |         type          | Description                                   |
| :----------------------------- | :-------------------: | :-------------------------------------------- |
| `ground_truth_objects`         | `List[DynamicObject]` | List of GT.                                   |
| `pointcloud_for_detection`     |    `numpy.ndarray`    | Pointcloud for detection area evaluation.     |
| `pointcloud_for_non_detection` | `List[numpy.ndarray]` | Pointcloud for non-detection area evaluation. |

## Evaluation per one Object

### `<class> DynamicObjectWithSensingResult(...)`

Sensing result for one GT.
For the details, see [perception_eval/result/sensing/sensing_result.py](../../../perception_eval/perception_eval/result/sensing/sensing_result.py)

| Argument                       |         type          | Description                                   |
| :----------------------------- | :-------------------: | :-------------------------------------------- |
| `ground_truth_objects`         | `List[DynamicObject]` | List of GT.                                   |
| `pointcloud_for_detection`     |    `numpy.ndarray`    | Pointcloud for detection area evaluation.     |
| `pointcloud_for_non_detection` | `List[numpy.ndarray]` | Pointcloud for non-detection area evaluation. |

| Attribute               |         type          | Description                                    |
| :---------------------- | :-------------------: | :--------------------------------------------- |
| `ground_truth_objects`  | `List[DynamicObject]` | List of GT.                                    |
| `inside_pointcloud`     |    `numpy.ndarray`    | Pointcloud included in bounding box.           |
| `inside_pointcloud_num` |         `int`         | Number of pointcloud included in bounding box. |
| `is_detected`           |        `bool`         | True if at least there is one point.           |
