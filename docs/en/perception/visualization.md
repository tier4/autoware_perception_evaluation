# [`<class> PerceptionVisualizer(...)`](../../../perception_eval/perception_eval/visualization/perception_visualizer.py)

- Visualization tool for perception result.

## How to use

### 1.Initialization

There are tree ways of initialization of `PerceptionVisualizer`, 1. Use `PerceptionVisualizationConfig` or 2. Use `PerceptionEvaluationConfig` or 3. Specify arguments directory.

```python
from perception_eval.visualization.perception_visualizer import PerceptionVisualizer
from perception_eval.visualization.perception_visualization_config import PerceptionVisualizationConfig
from perception_eval.config.perception_evaluation_config import PerceptionEvaluationConfig

# Pattern.1 : Load from PerceptionVisualizationConfig
config = PerceptionVisualizationConfig(...)
visualizer = PerceptionVisualizer(config)


# Pattern.2 : Load from PerceptionEvaluationConfig
config = PerceptionEvaluationConfig(...)
visualizer = PerceptionVisualizer.from_eval_cfg(config)


# Pattern.3 : Load from arguments. These are same with PerceptionVisualizationConfig's
visualizer = PerceptionVisualizer.from_args(...)
```

### 2. Visualization

- Visualize 1 frame

```python
# Visualize one frame to visualization_directory_path
# REQUIRED: frame_result: PerceptionFrameResult
visualizer.visualize_frame(frame_result)
```

- Visualize all frames (=scene)

```python
# Visualize all frames to visualization_directory_path
# REQUIRED: frame_results: List[PerceptionFrameResult]

visualizer.visualize_all(frame_results)
```

Each visualized image will be saved in `visualization_directory_path` specified in `PerceptionVisualizationConfig`

### `<class> PerceptionVisualizationConfig(...)`

| Arguments                      |         type          | Mandatory | Description                                                  |
| :----------------------------- | :-------------------: | :-------: | :----------------------------------------------------------- |
| `visualization_directory_path` |         `str`         |    Yes    | Directory path to save visualized images.                    |
| `frame_id`                     |         `str`         |    Yes    | Frame ID.(`base_link` or `map`)                              |
| `evaluation_task`              |   `EvaluationTask`    |    Yes    | Perception evaluation task.                                  |
| `height`                       |         `int`         |    No     | Height of image.                                             |
| `width`                        |         `int`         |    No     | width of image.                                              |
| `target_labels`                | `List[AutowareLabel]` |    No     | List of target labels.                                       |
| `max_x_position_list`          |     `List[float]`     |    No     | Maximum x position of evaluation area.                       |
| `max_y_position_list`          |     `List[float]`     |    No     | Maximum y position of evaluation area.                       |
| `max_distance_list`            |     `List[float]`     |    No     | Maximum distance of evaluation area.                         |
| `min_distance_list`            |     `List[float]`     |    No     | Minimum distance of evaluation area.                         |
| `min_point_numbers`            |      `List[int]`      |    No     | Minimum number of pointcloud included in bounding box of GT. |
| `confidence_threshold_list`    |     `List[float]`     |    No     | Threshold list of estimation's confidence.                   |
| `target_uuids`                 |      `List[str]`      |    No     | List of target GT's uuid.                                    |

## Known issues / Limitations

- `PerceptionVisualizer()` only supports 3D evaluation.
