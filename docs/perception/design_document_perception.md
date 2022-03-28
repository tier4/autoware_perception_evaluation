## ソフトウェアについて

- Evaluator
  - cloud 上にある評価系システムのこと
  - Lsim 評価を含めた評価アプリケーションの管理が行われている
- [logsim](https://github.com/tier4/logsim)
  - 評価を行う ros package のこと
  - [Perception logsim](https://github.com/tier4/logsim/blob/ros2/logsim/scripts/perception_evaluator_node.py)
- AWMLevaluation: Perception 評価用リポジトリの名前
  - EvaluationManager: 評価の計算等を行う class

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

## データ構造

### 評価単位でのデータ構造

- Catalog: Scenario の塊
  - 例：右折 UseCase = (UC-0001, UC-0003, UC-0006)

```yaml
- Catalog
  - List[Scenario]
```

- Scenario: 評価する単位
  - Scenario には 2 つ種類がある
  - DB-Scenario: DataBase 用 Dataset で主に学習用に用いる
    - 1 Scenario = DB-Scenario, n scene
  - UC-Scenario: UseCase 評価用 Dataset
    - 1 Scenario = UC-Scenario, 1 scene

```yaml
- Scenario
  - List[Scene]
```

- Scene: 1 連続 rosbag の単位
  - 1 rosbag から構成されるデータ群
  - 1 rosbag + pcd + jpeg + annotation の塊

```yaml
- Scene
  - List[PerceptionFrameResult]
```

### Frame 単位でのデータ構造

- PerceptionFrameResult: 1 pointcloud の入力と、その入力に対しての結果のまとまり
  - 入力データ: 1 pointcloud + List[ground truth objects]
  - object_results (List[DynamicObjectWithPerceptionResult]): Object ごとの評価
  - metrics_score (MetricsScore): Metrics 評価
  - pass_fail_result (PassFailResult): Use case 評価の結果
    - fp_objects (List[DynamicObjectWithPerceptionResult]): Use case 評価で FP (False Positive) の ObjectResult
    - fn_objects (List[DynamicObject]): Use case 評価で FN (False Negative) の DynamicObject
- 詳細は<https://github.com/tier4/AWMLevaluation/blob/develop/awml_evaluation/awml_evaluation/evaluation/result/perception_frame_result.py>

```yaml
[2022-01-11 10:37:00,854] [INFO] [lsim.py:146 <module>] Frame result example (frame_results[0]):
{'frame_name': '0',
 'ground_truth_objects': ' --- length of element 98 ---,',
 'metrics_score': {'config': {'evaluation_tasks': ['EvaluationTask.DETECTION'],
                              'map_thresholds_center_distance': [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
                              'map_thresholds_iou_3d': [[0.5, 0.5, 0.5, 0.5]],
                              'map_thresholds_iou_bev': [[0.5, 0.5, 0.5, 0.5]],
                              'map_thresholds_plane_distance': [[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
                              'max_x_position_list': [102.4, 102.4, 102.4, 102.4],
                              'max_y_position_list': [102.4, 102.4, 102.4, 102.4],
                              'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE',
                                                'AutowareLabel.PEDESTRIAN', 'AutowareLabel.MOTORBIKE']},
                   'maps': ' --- length of element 6 ---,'},
 'object_results': ' --- length of element 97 ---,',
 'pass_fail_result': {'critical_ground_truth_objects': ' --- length of element 23 ---,',
                      'critical_object_filter_config': {'max_x_position_list': [30.0, 30.0, 30.0, 30.0],
                                                        'max_y_position_list': [30.0, 30.0, 30.0, 30.0],
                                                        'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE',
                                                                          'AutowareLabel.PEDESTRIAN',
                                                                          'AutowareLabel.MOTORBIKE']},
                      'fn_objects': ' --- length of element 6 ---,',
                      'fp_objects_result': ' --- length of element 16 ---,',
                      'frame_pass_fail_config': {'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE',
                                                                   'AutowareLabel.PEDESTRIAN',
                                                                   'AutowareLabel.MOTORBIKE'],
                                                 'threshold_plane_distance_list': [2.0, 2.0, 2.0, 2.0]}},
 'pointcloud': None,
 'unix_time': 1624164470849887}
```

### Object 単位でのデータ構造

- DynamicObjectWithPerceptionResult: 1 predicted object（認識の推論結果の bounding box）に対しての結果
  - ground_truth_object: Ground truth
  - predicted_object: (Autoware の) 認識の推論結果
  - metrics_score: 評価数値
- List[DynamicObjectWithPerceptionResult]
  - predicted_object (DynamicObject): 推論結果の Object
  - ground_truth_object (Optional[DynamicObject]): Ground truth
  - center_distance (CenterDistanceMatching): 中心間距離
  - plane_distance (PlaneDistanceMatching): 面距離
    - NN の 2 点座標も入っている
  - iou_bev (IOUBEVMatching): BEV の IoU
  - iou_3d (IOU3dMatching): 3d の IoU
- 関数を叩く必要があるもの (DynamicObjectWithPerceptionResult method)
  - 結果: <https://github.com/tier4/AWMLevaluation/blob/develop/awml_evaluation/awml_evaluation/evaluation/result/object_result.py#L77>
- 関数を叩く必要があるもの (DynamicObject method)
  - 推論結果なら object_result.predicted_object.get_footprint() などになる
  - 4 隅座標 <https://github.com/tier4/AWMLevaluation/blob/develop/awml_evaluation/awml_evaluation/common/object.py#L198>
  - yaw 角 <https://github.com/tier4/AWMLevaluation/blob/develop/awml_evaluation/awml_evaluation/common/object.py#L185>

```yaml
[2022-01-11 13:30:22,041] [INFO] [lsim.py:150 <module>] Object result example (frame_results[0].object_results[0]):
{'center_distance': {'mode': 'MatchingMode.CENTERDISTANCE', 'value': 2.3086792761230366},
 'ground_truth_object': {'pointcloud_num': None,
                         'predicted_confidence': None,
                         'predicted_path': None,
                         'semantic_label': 'AutowareLabel.PEDESTRIAN',
                         'semantic_score': 1.0,
                         'state': {'orientation': {'q': [0.9999188233118326, 0.011923246709992032,
                                                         0.0014331172038332752, -0.00425783391558715]},
                                   'position': [51.36128164198862, 13.387720045757435, 0.9432761344718263],
                                   'size': [0.94, 1.135, 1.879],
                                   'velocity': [nan, nan, nan]},
                         'tracked_path': None,
                         'unix_time': 1624164470849887,
                         'uuid': '5e5edc8ab20a8d6b13920ace32e93efe'},
 'iou_3d': {'mode': 'MatchingMode.IOU3D', 'value': 0.0},
 'iou_bev': {'mode': 'MatchingMode.IOUBEV', 'value': 0.0},
 'is_label_correct': True,
 'plane_distance': {'ground_truth_nn_plane': [[50.88644897057802, 12.824387955085653, 0.9311460157404472],
                                              [50.896152257391414, 13.959024090062458, 0.9581957371792247]],
                    'mode': 'MatchingMode.PLANEDISTANCE',
                    'predicted_nn_plane': [[53.148849138551, 12.868266796561109, 1.0406238022039516],
                                           [53.23667235571584, 13.974898952448491, 1.2770077146913759]],
                    'value': 2.3020280422585797},
 'predicted_object': {'pointcloud_num': None,
                      'predicted_confidence': None,
                      'predicted_path': None,
                      'semantic_label': 'AutowareLabel.PEDESTRIAN',
                      'semantic_score': 0.46922582015218073,
                      'state': {'orientation': {'q': [0.993651362878679, 0.10527797935871155, 0.012653909381690113,
                                                      -0.037595141825116654]},
                                'position': [53.661281641988616, 13.387720045757435, 1.1432761344718263],
                                'size': [0.94, 1.135, 1.879],
                                'velocity': [nan, nan, nan]},
                      'tracked_path': None,
                      'unix_time': 1624164470849887,
                      'uuid': '5e5edc8ab20a8d6b13920ace32e93efe'}}
```

### Metrics

- Metrics score
  - config (MetricsScoreConfig): threshold が入っている

```yaml
[2022-01-06 17:38:57,086] [INFO] [func]  [lsim.py:151 <module>] Metrics example (final_metric_score):
{'config': {'evaluation_tasks': ['EvaluationTask.DETECTION'],
            'map_thresholds_center_distance': [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
            'map_thresholds_iou_3d': [[0.5, 0.5, 0.5, 0.5]],
            'map_thresholds_iou_bev': [[0.5, 0.5, 0.5, 0.5]],
            'map_thresholds_plane_distance': [[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
            'max_x_position_list': [102.4, 102.4, 102.4, 102.4],
            'max_y_position_list': [102.4, 102.4, 102.4, 102.4],
            'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE', 'AutowareLabel.PEDESTRIAN',
                              'AutowareLabel.MOTORBIKE']},
 'maps': ' --- length of element 6 ---,'}
```

- map (Map): mAP の score を計算した class
  - aphs (List[Ap]): 各 label の Ap
  - map (float): mAP の score

```yaml

[2022-01-06 17:38:57,087] [INFO] [func] [lsim.py:155 <module>] metrics example (final_metric_score.maps[0].aps[0]):
{'aphs': [{'ap': 0.8771527141490844,
'fp_list': ' --- length of element 3774 ---,',
'ground_truth_objects_num': 3770,
'matching_average': 0.3605163907481991,
'matching_mode': 'MatchingMode.CENTERDISTANCE',
'matching_standard_deviation': 0.0014330986556561386,
'matching_threshold_list': [1.0],
'objects_results_num': 3774,
'target_labels': ['AutowareLabel.CAR'],
'tp_list': ' --- length of element 3774 ---,',
'tp_metrics': {'mode': 'TPMetricsAph'}},
{'ap': 0.8727832485980411,
'fp_list': ' --- length of element 1319 ---,',
'ground_truth_objects_num': 1319,
'matching_average': 0.3605163907481991,
'matching_mode': 'MatchingMode.CENTERDISTANCE',
'matching_standard_deviation': 0.0014330986556561386,
'matching_threshold_list': [1.0],
'objects_results_num': 1319,
'target_labels': ['AutowareLabel.BICYCLE'],
'tp_list': ' --- length of element 1319 ---,',
'tp_metrics': {'mode': 'TPMetricsAph'}},
{'ap': 0.8735186482000276,
'fp_list': ' --- length of element 7658 ---,',
'ground_truth_objects_num': 7657,
'matching_average': 0.3605163907481991,
'matching_mode': 'MatchingMode.CENTERDISTANCE',
'matching_standard_deviation': 0.0014330986556561386,
'matching_threshold_list': [1.0],
'objects_results_num': 7658,
'target_labels': ['AutowareLabel.PEDESTRIAN'],
'tp_list': ' --- length of element 7658 ---,',
'tp_metrics': {'mode': 'TPMetricsAph'}},
{'ap': 0.8767288928721811,
'fp_list': ' --- length of element 335 ---,',
'ground_truth_objects_num': 335,
'matching_average': 0.3605163907481991,
'matching_mode': 'MatchingMode.CENTERDISTANCE',
'matching_standard_deviation': 0.0014330986556561386,
'matching_threshold_list': [1.0],
'objects_results_num': 335,
'target_labels': ['AutowareLabel.MOTORBIKE'],
'tp_list': ' --- length of element 335 ---,',
'tp_metrics': {'mode': 'TPMetricsAph'}}],
'aps': [{'ap': 1.0010610079575597,
'fp_list': ' --- length of element 3774 ---,',
'ground_truth_objects_num': 3770,
'matching_average': 0.3605163907481991,
'matching_mode': 'MatchingMode.CENTERDISTANCE',
'matching_standard_deviation': 0.0014330986556561386,
'matching_threshold_list': [1.0],
'objects_results_num': 3774,
'target_labels': ['AutowareLabel.CAR'],
'tp_list': ' --- length of element 3774 ---,',
'tp_metrics': {'mode': 'TPMetricsAp'}},
{'ap': 1.0,
'fp_list': ' --- length of element 1319 ---,',
'ground_truth_objects_num': 1319,
'matching_average': 0.3605163907481991,
'matching_mode': 'MatchingMode.CENTERDISTANCE',
'matching_standard_deviation': 0.0014330986556561386,
'matching_threshold_list': [1.0],
'objects_results_num': 1319,
'target_labels': ['AutowareLabel.BICYCLE'],
'tp_list': ' --- length of element 1319 ---,',
'tp_metrics': {'mode': 'TPMetricsAp'}},
{'ap': 1.0001305994514822,
'fp_list': ' --- length of element 7658 ---,',
'ground_truth_objects_num': 7657,
'matching_average': 0.3605163907481991,
'matching_mode': 'MatchingMode.CENTERDISTANCE',
'matching_standard_deviation': 0.0014330986556561386,
'matching_threshold_list': [1.0],
'objects_results_num': 7658,
'target_labels': ['AutowareLabel.PEDESTRIAN'],
'tp_list': ' --- length of element 7658 ---,',
'tp_metrics': {'mode': 'TPMetricsAp'}},
{'ap': 1.0,
'fp_list': ' --- length of element 335 ---,',
'ground_truth_objects_num': 335,
'matching_average': 0.3605163907481991,
'matching_mode': 'MatchingMode.CENTERDISTANCE',
'matching_standard_deviation': 0.0014330986556561386,
'matching_threshold_list': [1.0],
'objects_results_num': 335,
'target_labels': ['AutowareLabel.MOTORBIKE'],
'tp_list': ' --- length of element 335 ---,',
'tp_metrics': {'mode': 'TPMetricsAp'}}],
'map': 1.0002979018522606,
'map_config': {'matching_mode': 'MatchingMode.CENTERDISTANCE',
'matching_threshold_list': [1.0, 1.0, 1.0, 1.0],
'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE', 'AutowareLabel.PEDESTRIAN',
'AutowareLabel.MOTORBIKE']},
'maph': 0.8750458759548335}

```

## 評価 Metrics

### Matching

- <https://github.com/tier4/AWMLevaluation/blob/fix/sensing_lsim/awml_evaluation/awml_evaluation/evaluation/matching/object_matching.py>
- Center Distance 3d
- IoU BEV
- IoU 3D
- Plane Distance

#### Plane distance

- メトリクスにおける TP/FP の判定において，Usecase 評価で Ground truth object と Predicted object の**自車近傍の 2 点の距離の RMS**を以って判定する．具体的には，

  1. GT と Det それぞれにおいて，footprint の端点のうち Ego から近い面(=2 点)を選択する．
  2. その面同士における 2 通りの端点のペアから，合計距離が短いペアを選択し，これを自車近傍の 2 点する．
  3. 各ペアの距離の 2 乗平均平方根をとり，これを\*自車近傍の 2 点の距離の RMS\*\*と呼ぶ．

- 例
  1. GT において，Ego から近い面として面 g3g4 を選択する．Det においては，面 d3d4 を選択する．
  2. 端点のペアは，(g3d3, g4d4)と(g3d4, g4d3)の 2 通りある．合計距離が短いペアを選択する．図例では，(g3d3, g4d4)を選択する．
  3. 自車近傍の 2 点の距離の RMS = sqrt ( ( g3d3^2 + g4d4^2 ) / 2 )
  - 詳しくは，`get_uc_plane_distance`関数を参照
  - 1 の背景：検出された物体の奥行きが不明瞭なので，確度の高い自車近傍の点を選択している．

![pipeline](figure/uc_plane_distance.svg)

- なぜか各 rosbag ごとに（crop_box_filter を変更させて record して）点群の最大距離が異なる -> 検出能力が変わっているので PerceptionEvaluationConfig を変えて評価

### TP Metrics

- <https://github.com/tier4/AWMLevaluation/blob/fix/sensing_lsim/awml_evaluation/awml_evaluation/evaluation/metrics/detection/tp_metrics.py>
- mAP
- mAPH
