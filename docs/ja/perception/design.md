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
  - [frame_results](https://github.com/tier4/AWMLevaluation/blob/9aff0bc8c572af46cc071e2d1082f8265ef27694/awml_evaluation/test/perception_lsim.py#L124)
  - [scene_metrics_result](https://github.com/tier4/AWMLevaluation/blob/9aff0bc8c572af46cc071e2d1082f8265ef27694/awml_evaluation/test/perception_lsim.py#L134)
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
  - [便利関数](../../../awml_evaluation/awml_evaluation/util/debug.py)

```yaml
dict_result = class_to_dict(self.frame_results)
json_result = json.dump(dict_result)
```

### 座標系と nuscenes-devkit について

| Autoware module                   | 座標系           | 使っている nuscenes-devkit |
| :-------------------------------- | :--------------- | :------------------------- |
| detection/objects                 | base link 座標系 | detection 用               |
| tracking/objects                  | map 座標系       | prediction 用(TBD)         |
| objects (-> prediction/objects ?) | map 座標系       | prediction 用(TBD)         |
| pointcloud                        | base link 座標系 | detection 用 + uuid 利用   |

### PerceptionEvaluationManager

- Perception 評価を実行する class
- PerceptionEvaluationConfig に従い，detection/tracking/prediction の評価を行う

#### PerceptionEvaluationConfig

- [awml_evaluation/config/perception_evaluation_config.py](../../../awml_evaluation/awml_evaluation/config/perception_evaluation_config.py)を参考

- `PerceptionEvaluationConfig`の引数は以下

  | Arguments                |       type       | Description                                                      |
  | :----------------------- | :--------------: | :--------------------------------------------------------------- |
  | `dataset_path`           |   `List[str]`    | データセットパス(List[]で指定するが複数データ対応については TBD) |
  | `frame_id`               |      `str`       | オブジェクトの座標系，`map` or `base_link`                       |
  | `does_use_pointcloud`    |      `bool`      | データセットから点群データをロードするかの flag                  |
  | `result_root_directory`  |      `str`       | 評価結果，ログ，可視化結果等を保存するディレクトリのパス         |
  | `evaluation_config_dict` | `Dict[str, Any]` | 評価パラメータ                                                   |

##### `evaluation_config_dict`

- `evaluation_config_dict (Dict[str, Any])`には，`evaluation_task`に detection/tracking/prediction を指定して各評価パラメータを設定する

  ```python
  evaluation_config_dict: [Dict[str, Any]] = {
    "evaluation_task": "detection"/"tracking"/"prediction",
    ...
  }
  ```

- 各評価パラメータは，**1.オブジェクトのフィルタ用の閾値**，**2.TP/FP/FN 判定用の閾値**の 2 種類に分類される．

  なお，

  - **1. `DynamicObjectWithPerceptionResult`生成時のオブジェクトのフィルタ用の閾値**

    | Arguments              |    type     |     Mandatory      | Description                                                                                                      |
    | :--------------------- | :---------: | :----------------: | :--------------------------------------------------------------------------------------------------------------- |
    | `target_labels`        | `List[str]` |        Yes         | 評価対象ラベル名                                                                                                 |
    | `max_x_position`       |   `float`   |         \*         | 評価対象 object の最大 x 位置                                                                                    |
    | `max_y_position`       |   `float`   |         \*         | 評価対象 object の最大 y 位置                                                                                    |
    | `max_distance`         |   `float`   |         \*         | 評価対象 object の base_link からの最大距離                                                                      |
    | `min_distance`         |   `float`   |         \*         | 評価対象 object の base_link からの最小距離                                                                      |
    | `min_point_numbers`    | `List[int]` | Yes (in Detection) | ground truth object における，bbox 内の最小点群数．`min_point_numbers=0` の場合は，全 ground truth object を評価 |
    | `confidence_threshold` |   `float`   |         No         | 評価対象の estimated object の confidence の閾値                                                                 |
    | `target_uuids`         | `List[str]` |         No         | 特定の ground truth のみに対して評価を行いたい場合，対象とする ground truth の UUID を指定する                   |

    \* **max_x/y_position**，**max/min_distance**についてはどちらか片方のみ指定する必要がある．

  - **2. メトリクス評価時の `DynamicObjectWithPerceptionResult`の TP/FP/FN 判定用の閾値**

    | Arguments                    |     type      | Mandatory | Description                  |
    | :--------------------------- | :-----------: | :-------: | :--------------------------- |
    | `center_distance_thresholds` | `List[float]` |    Yes    | 中心間距離マッチング時の閾値 |
    | `plane_distance_thresholds`  | `List[float]` |    Yes    | 平面距離マッチング時の閾値   |
    | `iou_bev_thresholds`         | `List[float]` |    Yes    | BEV IoU 　マッチング時の閾値 |
    | `iou_3d_thresholds`          | `List[float]` |    Yes    | 3D IoU マッチング時の閾値    |

- **パラメータ指定時の Error ケース**

  - **1. `evaluation_task`に `detection/tracking/prediction` 以外を指定した場合**

  ```python
  evaluation_config_dict = {
    "evaluation_task": "foo",  # <-- fooを指定
    "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
    "max_x_position": 102.4,
    "max_y_position": 102.4,
    "min_point_numbers": [0, 0, 0, 0],
    "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
    "plane_distance_thresholds": [2.0, 3.0],
    "iou_bev_thresholds": [0.5],
    "iou_3d_thresholds": [0.5],
  }
  ```

  ```shell
  # Exception
  >>  ValueError: Unsupported task: 'foo'
      Supported tasks: ['detection', 'tracking', 'prediction']
  ```

  - **2. `max_x/y_position`，`max/min_distance` がどちらも未指定 or 両方指定した場合**

    - 両方指定

    ```python
    evaluation_config_dict = {
      "evaluation_task": "detection",
      "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
      "max_x_position": 102.4,
      "max_y_position": 102.4,
      "max_distance": 100.0,
      "min_distance": 10.0,
      "min_point_numbers": [0, 0, 0, 0],
      "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "plane_distance_thresholds": [2.0, 3.0],
      "iou_bev_thresholds": [0.5],
      "iou_3d_thresholds": [0.5],
    }
    ```

    ```shell
    # Exception
    >> RuntimeError: Either max x/y position or max/min distance should be specified
    ```

    - 両方未指定

    ```python
    evaluation_config_dict = {
      "evaluation_task": "detection",
      "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
      # "max_x_position": 102.4,  # <-- それぞれコメントアウト
      # "max_y_position": 102.4,
      # "max_distance": 100.0,
      # "min_distance": 10.0,
      "min_point_numbers": [0, 0, 0, 0],
      "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "plane_distance_thresholds": [2.0, 3.0],
      "iou_bev_thresholds": [0.5],
      "iou_3d_thresholds": [0.5],
    }
    ```

    ```shell
    # Exception
    >> RuntimeError: Either max x/y position or max/min distance should be specified
    ```

  - **3. Detection 評価時に`min_point_numbers`が未指定な場合**

    ```python
    evaluation_config_dict = {
      "evaluation_task": "detection",  # <-- detectionを指定
      "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
      "max_x_position": 102.4,
      "max_y_position": 102.4,
      # "min_point_numbers": [0, 0, 0, 0],  # <-- min_point_numbersをコメントアウト
      "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "plane_distance_thresholds": [2.0, 3.0],
      "iou_bev_thresholds": [0.5],
      "iou_3d_thresholds": [0.5],
    }
    ```

    ```shell
    # Exception
    >> RuntimeError: In detection task, min point numbers must be specified
    ```

  - **4. TP/FP/FN 判定用のパラメータに各 `MetricsConfig` の引数以外ものを設定した場合**

    - 各 MetricsConfig は，[awml_evaluation/evaluation/metrics/config](../../../awml_evaluation/awml_evaluation/evaluation/metrics/config/)を参考

    ```python
    evaluation_config_dict = {
      "evaluation_task": "detection",
      "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
      "max_x_position": 102.4,
      "max_y_position": 102.4,
      "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "plane_distance_thresholds": [2.0, 3.0],
      "iou_bev_thresholds": [0.5],
      "iou_3d_thresholds": [0.5],
      "min_point_numbers": [0, 0, 0, 0],
      "foo_thresholds": [0.8],  # <-- foo_thresholdsを指定
    }
    ```

    ```shell
    # Exception
    >>  awml_evaluation.evaluation.metrics.metrics_score_config.MetricsParameterError: MetricsConfig for 'EvaluationTask.DETECTION'
        Unexpected parameters: {'foo_thresholds'}
        Usage: {'plane_distance_thresholds', 'iou_3d_thresholds', 'center_distance_thresholds', 'target_labels', 'iou_bev_thresholds'}
    ```

### CriticalObjectFilterConfig

- 注目物体を動的決定するためのインターフェイス．
- `PerceptionEvaluationManger`の初期化時ではなく，各フレーム毎(=callback)に指定する．
- [awml_evaluation/evaluation/result/perception_frame_config](../../../awml_evaluation/awml_evaluation/evaluation/result/perception_frame_config.py)を参考

| Arguments                   |             type             |    Mandatory    | Description                                                                                                      |
| :-------------------------- | :--------------------------: | :-------------: | :--------------------------------------------------------------------------------------------------------------- |
| `evaluator_config`          | `PerceptionEvaluationConfig` |       Yes       | `PerceptionEvaluationManager`の持つ config                                                                       |
| `target_labels`             |         `List[str]`          |       Yes       | 評価対象ラベル名                                                                                                 |
| `max_x_position_list`       |        `List[float]`         |       \*        | 評価対象 object の最大 x 位置                                                                                    |
| `max_y_position_list`       |        `List[float]`         |       \*        | 評価対象 object の最大 y 位置                                                                                    |
| `max_distance_list`         |        `List[float]`         |       \*        | 評価対象 object の base_link からの最大距離                                                                      |
| `min_distance_list`         |        `List[float]`         |       \*        | 評価対象 object の base_link からの最小距離                                                                      |
| `min_point_numbers`         |         `List[int]`          | Yes (Detection) | ground truth object における，bbox 内の最小点群数．`min_point_numbers=0` の場合は，全 ground truth object を評価 |
| `confidence_threshold_list` |        `List[float]`         |       No        | 評価対象の estimated object の confidence の閾値                                                                 |
| `target_uuids`              |         `List[str]`          |       No        | 特定の ground truth のみに対して評価を行いたい場合，対象とする ground truth の UUID を指定する                   |

\* **max_x/y_position**，**max/min_distance**についてはどちらか片方のみ指定する必要がある．

### PerceptionPassFailConfig

- Pass / Fail を決めるためのパラメータ. Pass/Fail の判定については，**Plane distance**によって TP/FP の判定を行う．
- `PerceptionEvaluationManger`の初期化時ではなく，各フレーム毎(=callback)に指定する．
- [awml_evaluation/evaluation/result/perception_frame_config](../../../awml_evaluation/awml_evaluation/evaluation/result/perception_frame_config.py)を参考

| Arguments                       |             type             | Mandatory | Description                                      |
| :------------------------------ | :--------------------------: | :-------: | :----------------------------------------------- |
| `evaluator_config`              | `PerceptionEvaluationConfig` |    Yes    | `PerceptionEvaluationManager`の持つ config       |
| `target_labels`                 |         `List[str]`          |    Yes    | 評価対象ラベル名                                 |
| `plane_distance_threshold_list` |        `List[float]`         |    Yes    | 平面距離マッチング時の閾値                       |
| `confidence_threshold_list`     |        `List[float]`         |    No     | 評価対象の estimated object の confidence の閾値 |

### 評価実行

- Frame ごとの評価 : `PerceptionEvaluationManager::add_frame_result()`
  - frame 毎に予測オブジェクトと Ground Truth オブジェクトから各評価指標のスコアを計算する．
- Scene(=全 Frame)の評価 : `PerceptionEvaluationManager()::get_scene_result()`
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
