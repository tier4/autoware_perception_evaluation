# Visualize sensing results

## [`<class> SensingVisualizer(...)`](../../../perception_eval/perception_eval/visualization/sensing_visualizer.py)

- Visualization tool for sensing result.

### How to use

#### 1.Initialization

There are tree ways of initialization of `SensingVisualizer`, 1. Use `SensingEvaluationConfig` or 2. Specify scenario file path(.yaml).

```python
from perception_eval.visualization import SensingVisualizer
from perception_eval.config import SensingEvaluationConfig


# Pattern.1 : Load from SensingEvaluationConfig
config = SensingEvaluationConfig(...)
visualizer = SensingVisualizer.from_eval_cfg(config)


# Pattern.2 : Load from scenario file.
visualizer = SensingVisualizer.from_scenario(
    result_root_directory: str,
    scenario_filepath: str,
)
```

#### 2. Visualization

- Visualize 1 frame

```python
# Visualize one frame to visualization_directory_path
# REQUIRED: frame_result: SensingFrameResult
visualizer.visualize_frame(frame_result)
```

- Visualize all frames (=scene)

```python
# Visualize all frames to visualization_directory_path
# REQUIRED: frame_results: List[SensingFrameResult]

visualizer.visualize_all(frame_results)
```

Each visualized image and video of whole scene will be saved in `visualization_directory_path`.
