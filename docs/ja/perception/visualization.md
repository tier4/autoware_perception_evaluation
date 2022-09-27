# Perception Visualizer

- Perception 結果の可視化ツール

## How to use

### 1.Initialization

`PerceptionVisualizer`の初期化方法には，1.`PerceptionVisualizationConfig`を使用，2.`PerceptionEvaluationConfig`を使用，3. 直接引数を指定の 3 パターンある．

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

`PerceptionVisualizationConfig`に指定した，`visualization_directory_path`に各フレームの可視化画像が保存される．

### PerceptionVisualizationConfig

| Arguments                      |         type          | Mandatory | Description                             |
| :----------------------------- | :-------------------: | :-------: | :-------------------------------------- |
| `visualization_directory_path` |         `str`         |    Yes    | 可視化結果の保存ディレクトリのパス      |
| `frame_id`                     |         `str`         |    Yes    | Frame ID (`base_link` or `map`)         |
| `evaluation_task`              |   `EvaluationTask`    |    Yes    | Perception 評価タスク                   |
| `height`                       |         `int`         |    No     | 可視化画像の height                     |
| `width`                        |         `int`         |    No     | 可視化画像の width                      |
| `target_labels`                | `List[AutowareLabel]` |    No     | 評価対象ラベル                          |
| `max_x_position_list`          |     `List[float]`     |    No     | 評価対象領域の最大 x 位置               |
| `max_y_position_list`          |     `List[float]`     |    No     | 評価対象領域の最大 y 位置               |
| `max_distance_list`            |     `List[float]`     |    No     | 評価対象領域の最大距離                  |
| `min_distance_list`            |     `List[float]`     |    No     | 評価対象領域の最小距離                  |
| `min_point_numbers`            |      `List[int]`      |    No     | 評価対象オブジェクト box 内の最小点群数 |
| `confidence_threshold_list`    |     `List[float]`     |    No     | 評価対象オブジェクトの confidence 閾値  |
| `target_uuids`                 |      `List[str]`      |    No     | 評価対象オブジェクトの GT の uuid       |
