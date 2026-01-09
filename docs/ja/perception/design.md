## ソフトウェアについて

- Evaluator
  - cloud 上にある評価系システムのこと
  - Lsim 評価を含めた評価アプリケーションの管理が行われている
- [driving_log_replayer](https://github.com/tier4/driving_log_replayer)
  - 評価を行う ros package のこと
- autoware_perception_evaluation: Perception/Sensing 評価用ツールパッケージのリポジトリ名
  - Perception/SensingEvaluationManager: 評価の計算等を行う class

## 評価について

### Driving log Replayer 側で出力するべき情報

- driving_log_replayer 側で出力するべき情報
  - [perception_eval/test/perception_lsim.py](../../../perception_eval/test/perception_lsim.py)を参考
- frame_results: 評価を再計算するのに必要な情報
  - rosbag, annotation -> 評価に必要な情報 -> Metrics 計算 という流れで、rosbag, annotation からもう一度評価することなく Metrics を再計算するのに必要な情報、という意味
  - Use Case としては scenario 評価がある
    - 例えば 10 個の rosbag を（クラウド上で）分散して driving_log_replayer を回して result(scene_result.json)を出す
    - その 10 個の scene_result.json を用いて scenario 評価として result.json を作る
    - 時に scene_result.json に必要な情報が frame_results
- scene_metrics_result: scene の評価情報
  - rosbag ごとの評価値を見て解析が可能
- json 化について
  - class object -> dict -> json が一番簡単なはず
  - [便利関数](../../../perception_eval/perception_eval/util/debug.py)

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

### [`<class> PerceptionEvaluationManager(...)`](../../../perception_eval/perception_eval/manager/perception_evaluation_manager.py)

- Perception 評価を実行する class
- PerceptionEvaluationConfig に従い，detection/tracking/prediction の評価を行う

#### [`<class> PerceptionEvaluationConfig(...)`](../../../perception_eval/perception_eval/config/perception_evaluation_config.py)

- `PerceptionEvaluationConfig`の引数は以下

  | Arguments                |            type             | Description                                                      |
  | :----------------------- | :-------------------------: | :--------------------------------------------------------------- |
  | `dataset_path`           |         `List[str]`         | データセットパス(List[]で指定するが複数データ対応については TBD) |
  | `frame_id`               | `Union[str, Sequence[str]]` | オブジェクトが従う FrameIDの文字列.                              |
  | `result_root_directory`  |            `str`            | 評価結果，ログ，可視化結果等を保存するディレクトリのパス         |
  | `evaluation_config_dict` |      `Dict[str, Any]`       | 評価パラメータ                                                   |
  | `load_raw_data`          |           `bool`            | データセットから点群/画像データをロードするか                    |

##### `evaluation_config_dict`

- `evaluation_config_dict (Dict[str, Any])`には，`evaluation_task`に detection/tracking/prediction または detection2d を指定して各評価パラメータを設定する

  ```python
  evaluation_config_dict: [Dict[str, Any]] = {
    "evaluation_task": "detection"/"tracking"/"prediction" or "detection2d/tracking2d/classification2d",
    ...
  }
  ```

- 各評価パラメータは，**1.オブジェクトのフィルタ用の閾値**, **2.ラベル設定用のパラメータ**, **3.TP/FP/FN 判定用の閾値**の 3 種類に分類される．

  なお，

  - **1. `DynamicObjectWithPerceptionResult`生成時のオブジェクトのフィルタ用の閾値**

    | Arguments              |     type      |       Mandatory       | Description                                                                                                                 |
    | :--------------------- | :-----------: | :-------------------: | :-------------------------------------------------------------------------------------------------------------------------- |
    | `target_labels`        |  `List[str]`  |          No           | 評価対象ラベル名．未指定の場合，全ラベルを対象に評価が実行される。                                                          |
    | `ignore_attributes`    |  `List[str]`  |          No           | 評価対象外となるラベル名のキーワードもしくはアトリビュート。未指定の場合、全ラベルを対象に評価が実行される。                |
    | `max_x_position`       |    `float`    |          \*           | 評価対象 object の最大 x 位置 (3D のみ)                                                                                     |
    | `max_y_position`       |    `float`    |          \*           | 評価対象 object の最大 y 位置 (3D のみ)                                                                                     |
    | `max_distance`         |    `float`    |          \*           | 評価対象 object の base_link からの最大距離 (3D のみ)                                                                       |
    | `min_distance`         |    `float`    |          \*           | 評価対象 object の base_link からの最小距離 (3D のみ)                                                                       |
    | `max_matchable_radii`  | `List[float]` |          No           | マッチング時に許容する最大距離 (Defaults: `None`)                                                                           |
    | `min_point_numbers`    |  `List[int]`  | No (Yes in Detection) | ground truth object における，bbox 内の最小点群数．`min_point_numbers=0` の場合は，全 ground truth object を評価．(3D のみ) |
    | `confidence_threshold` |    `float`    |          No           | 評価対象の estimated object の confidence の閾値                                                                            |
    | `target_uuids`         |  `List[str]`  |          No           | 特定の ground truth のみに対して評価を行いたい場合，対象とする ground truth の UUID を指定する                              |

    \* **max_x/y_position**，**max/min_distance**についてはどちらか片方のみ指定する必要がある．

  - **3. ラベル設定用のパラメータ**

    | Arguments                     |  type  | Mandatory | Description                                                                                                  |
    | :---------------------------- | :----: | :-------: | :----------------------------------------------------------------------------------------------------------- |
    | `label_prefix`                | `str`  |    Yes    | 使用ラベル種類("autoware", "traffic_light")                                                                  |
    | `merge_similar_labels`        | `bool` |    No     | 類似ラベルをマージするか(Default: `False`)                                                                   |
    | `allow_matching_unknown`      | `bool` |    No     | unknownラベル予測と正解物体とのマージを許容するか(Default: `False`)                                          |
    | `count_label_number`          | `bool` |    No     | ロードされた各ラベルの数を数えるか(Default: `True`)                                                          |
    | `matching_class_agnostic_fps` | `bool` |    No     | False Positive を別のラベルにマッチさせる（クラス非依存マッチング）場合は True に設定する (Default: `False`) |

  - **2. メトリクス評価時の `DynamicObjectWithPerceptionResult`の TP/FP/FN 判定用の閾値**

    | Arguments                        |     type      | Mandatory | Description                          |
    | :------------------------------- | :-----------: | :-------: | :----------------------------------- |
    | `center_distance_thresholds`     | `List[float]` |    Yes    | 中心間距離マッチング時の閾値         |
    | `center_distance_bev_thresholds` | `List[float]` |    Yes    | 中心のBEV距離マッチング時の閾値      |
    | `plane_distance_thresholds`      | `List[float]` | Yes (3D)  | 平面距離マッチング時の閾値 (3D のみ) |
    | `iou_2d_thresholds`              | `List[float]` |    Yes    | BEV IoU 　マッチング時の閾値         |
    | `iou_3d_thresholds`              | `List[float]` | Yes (3D)  | 3D IoU マッチング時の閾値 (3D のみ)  |

- **パラメータ指定時の Error ケース**

  - **1. `evaluation_task`に `detection/tracking/prediction/detection2d/tracking2d/classification2d` 以外を指定した場合**

  ```python
  evaluation_config_dict = {
    "evaluation_task": "foo",  # <-- fooを指定
    "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
    "ignore_attributes": [],
    "max_x_position": 102.4,
    "max_y_position": 102.4,
    "min_point_numbers": [0, 0, 0, 0],
    "label_prefix": "autoware",
    "merge_similar_labels": False,
    "allow_matching_unknown": True,
    "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
    "center_distance_bev_thresholds": [[1.0, 1.0, 1.0, 1.0]],
    "plane_distance_thresholds": [2.0, 3.0],
    "iou_2d_thresholds": [0.5],
    "iou_3d_thresholds": [0.5],
  }
  ```

  ```shell
  # Exception
  >>  ValueError: Unsupported task: 'foo'
      Supported tasks: ['detection', 'tracking', 'prediction', 'detection2d', 'tracking2d', 'classification2d']
  ```

  - **2. `max_x/y_position`，`max/min_distance` がどちらも未指定 or 両方指定した場合**

    - 両方指定

    ```python
    evaluation_config_dict = {
      "evaluation_task": "detection",
      "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
      "ignore_attributes": [],
      "max_x_position": 102.4,
      "max_y_position": 102.4,
      "max_distance": 100.0,
      "min_distance": 10.0,
      "min_point_numbers": [0, 0, 0, 0],
      "label_prefix": "autoware",
      "label_prefix": "autoware",
      "merge_similar_labels": False,
      "allow_matching_unknown": True,
      "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "center_distance_bev_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "plane_distance_thresholds": [2.0, 3.0],
      "iou_2d_thresholds": [0.5],
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
      "ignore_attributes": [],
      # "max_x_position": 102.4,  # <-- それぞれコメントアウト
      # "max_y_position": 102.4,
      # "max_distance": 100.0,
      # "min_distance": 10.0,
      "min_point_numbers": [0, 0, 0, 0],
      "label_prefix": "autoware",
      "merge_similar_labels": False,
      "allow_matching_unknown": True,
      "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "center_distance_bev_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "plane_distance_thresholds": [2.0, 3.0],
      "iou_2d_thresholds": [0.5],
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
      "ignore_attributes": [],
      "max_x_position": 102.4,
      "max_y_position": 102.4,
      # "min_point_numbers": [0, 0, 0, 0],  # <-- min_point_numbersをコメントアウト
      "label_prefix": "autoware",
      "merge_similar_labels": False,
      "allow_matching_unknown": True,
      "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "center_distance_bev_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "plane_distance_thresholds": [2.0, 3.0],
      "iou_2d_thresholds": [0.5],
      "iou_3d_thresholds": [0.5],
    }
    ```

    ```shell
    # Exception
    >> RuntimeError: In detection task, min point numbers must be specified
    ```

  - **4. TP/FP/FN 判定用のパラメータに各 `MetricsConfig` の引数以外ものを設定した場合**

    - 各 MetricsConfig は，[perception_eval/evaluation/metrics/config](../../../perception_eval/perception_eval/evaluation/metrics/config/)を参考

    ```python
    evaluation_config_dict = {
      "evaluation_task": "detection",
      "target_labels": ["car", "bicycle", "pedestrian", "motorbike"],
      "ignore_attributes": [],
      "max_x_position": 102.4,
      "max_y_position": 102.4,
      "min_point_numbers": [0, 0, 0, 0],
      "label_prefix": "autoware",
      "merge_similar_labels": False,
      "allow_matching_unknown": True,
      "center_distance_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "center_distance_bev_thresholds": [[1.0, 1.0, 1.0, 1.0]],
      "plane_distance_thresholds": [2.0, 3.0],
      "iou_2d_thresholds": [0.5],
      "iou_3d_thresholds": [0.5],
      "min_point_numbers": [0, 0, 0, 0],
      "foo_thresholds": [0.8],  # <-- foo_thresholdsを指定
    }
    ```

    ```shell
    # Exception
    >>  perception_eval.evaluation.metrics.metrics_score_config.MetricsParameterError: MetricsConfig for 'EvaluationTask.DETECTION'
        Unexpected parameters: {'foo_thresholds'}
        Usage: {'plane_distance_thresholds', 'iou_3d_thresholds', 'center_distance_thresholds', 'target_labels', 'iou_bev_thresholds'}
    ```

### [`<class> CriticalObjectFilterConfig(...)`](../../../perception_eval/perception_eval/evaluation/result/perception_frame_config.py)

- 注目物体を動的決定するためのインターフェイス．
- `PerceptionEvaluationManager`の初期化時ではなく，各フレーム毎(=callback)に指定する．

| Arguments                   |             type             |    Mandatory    | Description                                                                                                                 |
| :-------------------------- | :--------------------------: | :-------------: | :-------------------------------------------------------------------------------------------------------------------------- |
| `evaluator_config`          | `PerceptionEvaluationConfig` |       Yes       | `PerceptionEvaluationManager`の持つ config                                                                                  |
| `target_labels`             |         `List[str]`          |       Yes       | 評価対象ラベル名                                                                                                            |
| `ignore_attributes`         |         `List[str]`          |       No        | 評価対象外となるラベル名のキーワードもしくはアトリビュート。                                                                |
| `max_x_position_list`       |        `List[float]`         |       \*        | 評価対象 object の最大 x 位置 (3D のみ)                                                                                     |
| `max_y_position_list`       |        `List[float]`         |       \*        | 評価対象 object の最大 y 位置 (3D のみ)                                                                                     |
| `max_distance_list`         |        `List[float]`         |       \*        | 評価対象 object の base_link からの最大距離 (3D のみ)                                                                       |
| `min_distance_list`         |        `List[float]`         |       \*        | 評価対象 object の base_link からの最小距離 (3D のみ)                                                                       |
| `min_point_numbers`         |         `List[int]`          | Yes (Detection) | ground truth object における，bbox 内の最小点群数．`min_point_numbers=0` の場合は，全 ground truth object を評価．(3D のみ) |
| `confidence_threshold_list` |        `List[float]`         |       No        | 評価対象の estimated object の confidence の閾値                                                                            |
| `target_uuids`              |         `List[str]`          |       No        | 特定の ground truth のみに対して評価を行いたい場合，対象とする ground truth の UUID を指定する                              |

\* **max_x/y_position**，**max/min_distance**についてはどちらか片方のみ指定する必要がある．

### [`<class> PerceptionPassFailConfig(...)`](../../../perception_eval/perception_eval/evaluation/result/perception_frame_config.py)

- Pass / Fail を決めるためのパラメータ. Pass/Fail の判定については，**Plane distance**によって TP/FP の判定を行う．
- `PerceptionEvaluationManager`の初期化時ではなく，各フレーム毎(=callback)に指定する．

| Arguments                   |             type             | Mandatory | Description                                            |
| :-------------------------- | :--------------------------: | :-------: | :----------------------------------------------------- |
| `evaluator_config`          | `PerceptionEvaluationConfig` |    Yes    | `PerceptionEvaluationManager`の持つ config             |
| `target_labels`             |         `List[str]`          |    No     | 評価対象ラベル名                                       |
| `matching_threshold_list`   |        `List[float]`         |    No     | マッチング閾値．3D の場合は平面距離，2D の場合は IOU． |
| `confidence_threshold_list` |        `List[float]`         |    No     | 評価対象の estimated object の confidence の閾値       |

### 評価実行

- Frame ごとの評価 : `PerceptionEvaluationManager::add_frame_result()`
  - frame 毎に予測オブジェクトと Ground Truth オブジェクトから各評価指標のスコアを計算する．
- Scene(=全 Frame)の評価 : `PerceptionEvaluationManager()::get_scene_result()`
  - scene(=全 frame)の各評価指標のスコアを計算する．

## データ構造

- 詳細については，[data_structure.md](./data_structure.md)を参照

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

### Classification

| Metrics  |             Sub Metrics              |
| :------: | :----------------------------------: |
| Accuracy | Accuracy, Precision, Recall, F1score |
