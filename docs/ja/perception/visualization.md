# Perception 結果の可視化

- 3D 評価: `PerceptionVisualizer3D`
- 2D 評価: `PerceptionVisualizer2D`

## [`<class> PerceptionVisualizer3D(...)`](../../../perception_eval/perception_eval/visualization/perception_visualizer3d.py)

- 3D Perception 結果の可視化ツール

### How to use

#### 1.Initialization

`PerceptionVisualizer3D`の初期化方法には，1.`PerceptionEvaluationConfig`を使用，2. シナリオファイル(.yaml)を使用の 2 パターンある．

```python
from perception_eval.visualization import PerceptionVisualizer3D
from perception_eval.config import PerceptionEvaluationConfig


# Pattern.1 : Load from PerceptionEvaluationConfig
config = PerceptionEvaluationConfig(...)
visualizer = PerceptionVisualizer3D(config)


# Pattern.2 : Load from scenario file
visualizer = PerceptionVisualizer3D.from_scenario(
    result_root_directory: str,
    scenario_file_path: str,
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

`visualization_directory_path`に各フレームの可視化画像と全フレーム分のが保存される．

## [`<class> PerceptionVisualizer2D(...)`](../../../perception_eval/perception_eval/visualization/perception_visualizer2d.py)

- 2D Perception 結果の可視化ツール

### How to use

#### 1.Initialization

`PerceptionVisualizer2D`の初期化方法には，1.`PerceptionEvaluationConfig`を使用，2. シナリオファイル(.yaml)を使用の 2 パターンある．

```python
from perception_eval.visualization import PerceptionVisualizer2D
from perception_eval.config import PerceptionEvaluationConfig


# Pattern.1 : Load from PerceptionEvaluationConfig
config = PerceptionEvaluationConfig(...)
visualizer = PerceptionVisualizer2D(config)


# Pattern.2 : Load from scenario file
visualizer = PerceptionVisualizer2D.from_scenario(
    result_root_directory: str,
    scenario_file_path: str,

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

`visualization_directory_path`に各フレームの可視化画像と全フレーム分のが保存される．
