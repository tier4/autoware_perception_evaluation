## ソフトウェアについて

- Evaluator
  - cloud 上にある評価系システムのこと
  - Lsim 評価を含めた評価アプリケーションの管理が行われている
- [logsim](https://github.com/tier4/logsim)
  - 評価を行う ros package のこと
  - [Perception logsim](https://github.com/tier4/logsim/blob/ros2/logsim/scripts/perception_evaluator_node.py)
- AWMLevaluation: Perception/Sensing 評価用ツールパッケージのリポジトリ名
  - Perception/SensingEvaluationManager: 評価の計算等を行う class

## 評価について

### logsim 側で出力するべき情報

- logsim 側で出力するべき情報
  - [frame_results](https://github.com/tier4/AWMLevaluation/blob/v1.2.0/awml_evaluation/awml_evaluation/perception_evaluation_manager.py#L46)
  - [scene_metrics_result](https://github.com/tier4/AWMLevaluation/blob/v1.2.0/awml_evaluation/test/lsim.py#L97)
- frame_results: 評価を再計算するのに必要な情報
  - rosbag, annotation -> 評価に必要な情報 -> Metrics 計算 という流れで、rosbag, annotation からもう一度評価することなく Metrics を再計算するのに必要な情報、という意味
  - Use Case としては scenario 評価がある
    - 例えば 10 個の rosbag を（クラウド上で）分散して logsim を回して result(scene_result.json)を出す
    - その 10 個の scene_result.json を用いて scenario 評価として result.json を作る
    - 時に scene_result.json に必要な情報が frame_results
- scene_metrics_result: scene の評価情報
  - rosbag ごとの評価値を見て解析が可能
- json 化について
  - class object -> dict -> json が一番簡単なはず
  - [便利関数](https://github.com/tier4/AWMLevaluation/blob/v1.2.0/awml_evaluation/awml_evaluation/util/debug.py#L35)

```yaml
dict_result = class_to_dict(self.frame_results)
json_result = json.dump(dict_result)
```

### 座標系と nuscenes-devkit について

| Autoware module                   | 座標系          | 使っている nuscenes-devkit |
| :-------------------------------- | :-------------- | :------------------------- |
| detection/objects                 | baselink 座標系 | detection 用               |
| tracking/objects                  | map 座標系      | prediction 用(TBD)         |
| objects (-> prediction/objects ?) | map 座標系      | prediction 用(TBD)         |
| pointcloud                        | baselink 座標系 | detection 用 + uuid 利用   |

### PerceptionEvaluationManager

- Perception 評価を実行する class
- PerceptionEvaluationConfig に従い，detection/tracking/prediction の評価を行う

  - [awml_evaluation/config/perception_evaluation_config.py](../../awml_evaluation/awml_evaluation/config/perception_evaluation_config.py)を参考

  - `evaluation_config_dict (Dict[str, Any])`には，`evaluation_task`に detection/tracking/prediction を指定して各評価パラメータを設定する

    ```python
    evaluation_config_dict: [Dict[str, Any]] = {
      "evaluation_task": "detection"/"tracking"/"prediction",
      ...
    }
    ```

    - detection
      - target_labels (List[str]): 評価対象ラベル名
      - max_x_position (float): 評価対象 object の最大 x 位置
      - max_y_position (float): 評価対象 object の最大 y 位置
      - center_distance_thresholds (List[float]): 中心間距離マッチング時の閾値
      - plane_distance_thresholds (List[float]): 平面距離マッチング時の閾値
      - iou_bev_thresholds (List[float]): BEV IoU 　マッチング時の閾値
      - iou_3d_thresholds (List[float]): 3D IoU マッチング時の閾値
    - tracking
      - target_labels (List[str]): 評価対象ラベル名
      - max_x_position (float): 評価対象 object の最大 x 位置
      - max_y_position (float): 評価対象 object の最大 y 位置
      - center_distance_thresholds (List[float]): 中心間距離マッチング時の閾値
      - plane_distance_thresholds (List[float]): 平面距離マッチング時の閾値
      - iou_bev_thresholds (List[float]): BEV IoU 　マッチング時の閾値
      - iou_3d_thresholds (List[float]): 3D IoU マッチング時の閾値
    - prediction (TBD)

  - `evaluation_task`に detection/tracking/prediction 以外を指定すると Error

    - 以下のように`hoge`とすると，

    ```python
    {
      "evaluation_task": "hoge",  # hogeを指定
      "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
      "max_x_position": 102.4,
      "max_y_position": 102.4,
      "center_distance_thresholds": [
          [1.0, 1.0, 1.0, 1.0],
          [2.0, 2.0, 2.0, 2.0],
      ],
      "plane_distance_thresholds": [2.0, 3.0],
      "iou_bev_thresholds": [0.5],
      "iou_3d_thresholds": [0.5],
    },
    ```

    - 以下のような Error メッセージが表示される

    ```shell
    Traceback (most recent call last):
    File "/usr/lib/python3.8/runpy.py", line 194, in _run_module_as_main
      return _run_code(code, main_globals, None,
    File "/usr/lib/python3.8/runpy.py", line 87, in _run_code
      exec(code, run_globals)
    File "/AWMLevaluation/awml_evaluation/test/lsim.py", line 264, in <module>
      perception_lsim = PerceptionLSimMoc(dataset_paths)
    File "/AWMLevaluation/awml_evaluation/test/lsim.py", line 59, in __init__
      evaluation_config: PerceptionEvaluationConfig = PerceptionEvaluationConfig(
    File "/AWMLevaluation/awml_evaluation/awml_evaluation/config/perception_evaluation_config.py", line 48, in __init__
      super().__init__(
    File "/AWMLevaluation/awml_evaluation/awml_evaluation/config/_evaluation_config_base.py", line 53, in __init__
      self._check_tasks(evaluation_config_dict)
    File "/AWMLevaluation/awml_evaluation/awml_evaluation/config/_evaluation_config_base.py", line 83, in _check_tasks
      raise ValueError(
    ValueError: Unsupported task: 'hoge'
    Supported tasks: ['detection', 'tracking', 'prediction']
    ```

  - 評価パラメータは各 MetricsConfig の引数以外ものを設定すると Error

    - 各 MetricsConfig は，[awml_evaluation/evaluation/metrics/config](../../awml_evaluation/awml_evaluation/evaluation/metrics/config/)を参考

    - 以下のように detection で`hogehoge_thresholds`を指定すると，

    ```python
    {
      "evaluation_task": "detection",
      "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
      "max_x_position": 102.4,
      "max_y_position": 102.4,
      "center_distance_thresholds": [
          [1.0, 1.0, 1.0, 1.0],
          [2.0, 2.0, 2.0, 2.0],
      ],
      "plane_distance_thresholds": [2.0, 3.0],
      "iou_bev_thresholds": [0.5],
      "iou_3d_thresholds": [0.5],
      "hogehoge_thresholds": [0.8],  # 追加
    }
    ```

    - 以下のような Error メッセージが表示される

    ```shell
    Traceback (most recent call last):
    File "/usr/lib/python3.8/runpy.py", line 194, in _run_module_as_main
      return _run_code(code, main_globals, None,
    File "/usr/lib/python3.8/runpy.py", line 87, in _run_code
      exec(code, run_globals)
    File "/AWMLevaluation/awml_evaluation/test/lsim.py", line 265, in <module>
      perception_lsim = PerceptionLSimMoc(dataset_paths)
    File "/AWMLevaluation/awml_evaluation/test/lsim.py", line 60, in __init__
      evaluation_config: PerceptionEvaluationConfig = PerceptionEvaluationConfig(
    File "/AWMLevaluation/awml_evaluation/awml_evaluation/config/perception_evaluation_config.py", line 68, in __init__
      self.metrics_config: MetricsScoreConfig = MetricsScoreConfig(
    File "/AWMLevaluation/awml_evaluation/awml_evaluation/evaluation/metrics/metrics_score_config.py", line 42, in __init__
      _check_parameters(DetectionMetricsConfig, item)
    File "/AWMLevaluation/awml_evaluation/awml_evaluation/evaluation/metrics/metrics_score_config.py", line 71, in _check_parameters
      raise MetricsParameterError(
    awml_evaluation.evaluation.metrics.metrics_score_config.MetricsParameterError: MetricsConfig for 'EvaluationTask.DETECTION'
    Unexpected parameters: {'hogehoge_thresholds'}
    Usage: {'plane_distance_thresholds', 'iou_3d_thresholds', 'center_distance_thresholds', 'target_labels', 'max_x_position', 'max_y_position', 'iou_bev_thresholds'}
    ```

- 評価実行
  - add_frame_result()
    - frame 毎に予測オブジェクトと Ground Truth オブジェクトから各評価指標のスコアを計算する．
  - get_scene_result()
    - scene(=全 frame)の各評価指標のスコアを計算する．

## データ構造

- 詳細については，[data_construction.md](./data_construction.md)を参照

## Metrics

- 詳細については，[metrics.md](./metrics.md)を参照

### detection

| Metrics | Sub Metrics |
| ------- | ----------- |
| mAP     | AP, APH     |

### tracking

|  Metrics   |      Sub Metrics      |
| :--------: | :-------------------: |
|   CLEAR    | MOTA, MOTP, ID switch |
| HOTA (TBD) |   HOTA, DetA, AssA    |

### Prediction (TBD)

| Metrics | Sub Metrics |
| :-----: | :---------: |
