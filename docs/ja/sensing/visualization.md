# Sensing 結果の可視化

## [`<class> SensingVisualizer(...)`](../../../perception_eval/perception_eval/visualization/sensing_visualizer.py)

- Sensing 結果の可視化ツール

### How to use

#### 1.Initialization

`SensingVisualizer`の初期化方法には，1.`SensingEvaluationConfig`を使用，2. シナリオファイル(.yaml)を使用の 2 パターンある．

```python
from perception_eval.visualization import SensingVisualizer
from perception_eval.config import SensingEvaluationConfig


# Pattern.1 : Load from SensingEvaluationConfig
config = SensingEvaluationConfig(...)
visualizer = SensingVisualizer(config)


# Pattern.2 : Load from scenario file
visualizer = SensingVisualizer.from_scenario(
    result_root_directory: str,
    scenario_file_path: str,
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

`visualization_directory_path`に各フレームの可視化画像と全フレーム分の動画が保存される．
