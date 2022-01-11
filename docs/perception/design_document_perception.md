
## ソフトウェアについて

- Evaluator
  - cloud上にある評価系システムのこと
  - Lsim評価を含めた評価アプリケーションの管理が行われている
- [logsim](https://github.com/tier4/logsim)
  - 評価を行うros packageのこと
  - [Perception logsim](https://github.com/tier4/logsim/blob/ros2/logsim/scripts/perception_evaluator_node.py)
- AWMLevaluation: Perception評価用リポジトリの名前
  - PerceptionEvaluationManager: 評価の計算等を行うclass

## 評価について

### 座標系とnuscenes-devkitについて

| Autoware module                   | 座標系         | 使っているnuscenes-devkit |
| :-------------------------------- | :------------- | :------------------------ |
| detection/objects                 | baselink座標系 | detection用               |
| tracking/objects                  | map座標系      | prediction用(TBD)         |
| objects (-> prediction/objects ?) | map座標系      | prediction用(TBD)         |
| pointcloud                        | baselink座標系 | detection用 + uuid利用    |

## データ構造

### 評価単位でのデータ構造

- Catalog: Scenarioの塊
  - 例：右折UseCase = (UC-0001, UC-0003, UC-0006)

```
- Catalog
  - List[Scenario]
```

- Scenario: 評価する単位
  - Scenarioには2つ種類がある
  - DB-Scenario: DataBase用Datasetで主に学習用に用いる
    - 1 Scenario = DB-Scenario, n scene
  - UC-Scenario: UseCase評価用Dataset
    - 1 Scenario = UC-Scenario, 1 scene

```
- Scenario
  - List[Scene]
```

- Scene: 1 連続 rosbagの単位
  - 1 rosbagから構成されるデータ群
  - 1 rosbag + pcd + jpeg + annotationの塊

```
- Scene
  - List[PerceptionFrameResult]
```

### Frame単位でのデータ構造

- PerceptionFrameResult: 1 pointcloudの入力と、その入力に対しての結果のまとまり
  - 入力データ: 1 pointcloud + List[ground truth objects]
  - object_results (List[DynamicObjectWithResult]): Objectごとの評価
  - metrics_score (MetricsScore): Metrics評価
  - pass_fail_result (PassFailResult): Use case評価の結果
    - fp_objects (List[DynamicObjectWithResult]): Use case評価でFP (False Positive) のObjectResult
    - fn_objects (List[DynamicObject]): Use case評価でFN (False Negative) のDynamicObject
- 詳細は<https://github.com/tier4/AWMLevaluation/blob/develop/awml_evaluation/awml_evaluation/evaluation/result/perception_frame_result.py>

```
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

### Object単位でのデータ構造

- DynamicObjectWithResult: 1 predicted object（認識の推論結果のbounding box）に対しての結果
  - ground_truth_object: Ground truth
  - predicted_object: (Autowareの) 認識の推論結果
  - metrics_score: 評価数値
- List[DynamicObjectWithResult]
  - predicted_object (DynamicObject): 推論結果のObject
  - ground_truth_object (Optional[DynamicObject]): Ground truth
  - center_distance (CenterDistanceMatching): 中心間距離
  - plane_distance (PlaneDistanceMatching): 面距離
    - NNの2点座標も入っている
  - iou_bev (IOUBEVMatching): BEVのIoU
  - iou_3d (IOU3dMatching): 3dのIoU
- 関数を叩く必要があるもの (DynamicObjectWithResult method)
  - 結果: <https://github.com/tier4/AWMLevaluation/blob/develop/awml_evaluation/awml_evaluation/evaluation/result/object_result.py#L77>
- 関数を叩く必要があるもの (DynamicObject method)
  - 推論結果なら object_result.predicted_object.get_footprint() などになる
  - 4隅座標 <https://github.com/tier4/AWMLevaluation/blob/develop/awml_evaluation/awml_evaluation/common/object.py#L198>
  - yaw角 <https://github.com/tier4/AWMLevaluation/blob/develop/awml_evaluation/awml_evaluation/common/object.py#L185>

```
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

### Metrics について

- Metrics score
  - config (MetricsScoreConfig): thresholdが入っている

```
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

- map (Map): mAPのscoreを計算したclass
  - aphs (List[Ap]): 各labelのAp
  - map (float): mAPのscore

```
[2022-01-06 17:38:57,087] [INFO] [func]  [lsim.py:155 <module>] metrics example (final_metric_score.maps[0].aps[0]): 
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
