# Perception 結果の可視化

- 3D 評価: `PerceptionVisualizer3D`
- 2D 評価: `PerceptionVisualizer2D`

## [`<class> PerceptionVisualizer3D(...)`](../../../perception_eval/perception_eval/visualization/perception_visualizer3d.py)

- Perception 結果の可視化ツール

### How to use

#### 1.Initialization

`PerceptionVisualizer3D`の初期化方法には，1.`PerceptionVisualizeConfig3D`を使用，2.`PerceptionEvaluationConfig`を使用，3. 直接引数を指定の 3 パターンある．

```python
from perception_eval.visualization import PerceptionVisualizer3D
from perception_eval.visualization import PerceptionVisualizationConfig3D
from perception_eval.config import PerceptionEvaluationConfig

# Pattern.1 : Load from PerceptionVisualizeConfig3D
config = PerceptionVisualizationConfig3D(...)
visualizer = PerceptionVisualizer3D(config)


# Pattern.2 : Load from PerceptionEvaluationConfig
config = PerceptionEvaluationConfig(...)
visualizer = PerceptionVisualizer3D.from_eval_cfg(config)


# Pattern.3 : Load from arguments. These are same with PerceptionVisualizeConfig3D's arguments
visualizer = PerceptionVisualizer3D.from_args(...)
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

`PerceptionVisualizeConfig3D`に指定した，`visualization_directory_path`に各フレームの可視化画像が保存される．

### `<class> PerceptionVisualizeConfig3D(...)`

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

## [`<class> PerceptionVisualizer2D(...)`](../../../perception_eval/perception_eval/visualization/perception_visualizer2d.py)

- Perception 結果の可視化ツール

### How to use

#### 1.Initialization

`PerceptionVisualizer2D`の初期化方法には，1.`PerceptionVisualizeConfig2D`を使用，2.`PerceptionEvaluationConfig`を使用，3. 直接引数を指定の 3 パターンある．

```python
from perception_eval.visualization import PerceptionVisualizer2D
from perception_eval.visualization import PerceptionVisualizationConfig2D
from perception_eval.config import PerceptionEvaluationConfig

# Pattern.1 : Load from PerceptionVisualizeConfig2D
config = PerceptionVisualizeConfig2D(...)
visualizer = PerceptionVisualizer2D(config)


# Pattern.2 : Load from PerceptionEvaluationConfig
config = PerceptionEvaluationConfig(...)
visualizer = PerceptionVisualizer2D.from_eval_cfg(config)


# Pattern.3 : Load from arguments. These are same with PerceptionVisualizeConfig2D's arguments
visualizer = PerceptionVisualizer2D.from_args(...)
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

`PerceptionVisualizeConfig2D`に指定した，`visualization_directory_path`に各フレームの可視化画像が保存される．

### `<class> PerceptionVisualizeConfig2D(...)`

| Arguments                      |         type          | Mandatory | Description                            |
| :----------------------------- | :-------------------: | :-------: | :------------------------------------- |
| `visualization_directory_path` |         `str`         |    Yes    | 可視化結果の保存ディレクトリのパス     |
| `evaluation_task`              |   `EvaluationTask`    |    Yes    | Perception 評価タスク                  |
| `height`                       |         `int`         |    No     | 可視化画像の height                    |
| `width`                        |         `int`         |    No     | 可視化画像の width                     |
| `target_labels`                | `List[AutowareLabel]` |    No     | 評価対象ラベル                         |
| `confidence_threshold_list`    |     `List[float]`     |    No     | 評価対象オブジェクトの confidence 閾値 |
| `target_uuids`                 |      `List[str]`      |    No     | 評価対象オブジェクトの GT の uuid      |
