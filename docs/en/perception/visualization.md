# Visualize perception results

- 3D visualization: `PerceptionVisualizer3D`
- 2D visualization: `PerceptionVisualizer2D`

## [`<class> PerceptionVisualizer3D(...)`](../../../perception_eval/perception_eval/visualization/perception_visualizer3d.py)

- Visualization tool for perception result.

### How to use

#### 1.Initialization

There are tree ways of initialization of `PerceptionVisualizer3D`, 1. Use `PerceptionEvaluationConfig` or 2. Specify scenario file path(.yaml).

```python
from perception_eval.visualization import PerceptionVisualizer3D
from perception_eval.config import PerceptionEvaluationConfig


# Pattern.1 : Load from PerceptionEvaluationConfig
config = PerceptionEvaluationConfig(...)
visualizer = PerceptionVisualizer3D.from_eval_cfg(config)


# Pattern.2 : Load from scenario file.
visualizer = PerceptionVisualizer3D.from_scenario(
    result_root_directory: str,
    scenario_filepath: str,
)
```

#### 2. Visualization

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

Each visualized image and video of whole scene will be saved in `visualization_directory_path`.

## [`<class> PerceptionVisualizer2D(...)`](../../../perception_eval/perception_eval/visualization/perception_visualizer2d.py)

### How to use

#### 1.Initialization

There are tree ways of initialization of `PerceptionVisualizer2D`, 1. Use `PerceptionEvaluationConfig` or 2. Specify scenario file path(.yaml).

```python
from perception_eval.visualization import PerceptionVisualizer2D
from perception_eval.config import PerceptionEvaluationConfig


# Pattern.1: Load from PerceptionEvaluationConfig
config = PerceptionEvaluationConfig(...)
visualizer = PerceptionVisualizer2D.from_eval_cfg(config)


# Pattern.2 : Load from scenario file.
visualizer = PerceptionVisualizer2D.from_scenario(
    result_root_directory: str,
    scenario_filepath: str,
)
```

#### 2. Visualization

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

Each visualized image and video of whole scene will be saved in `visualization_directory_path`.
