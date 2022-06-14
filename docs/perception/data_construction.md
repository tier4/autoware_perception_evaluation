# Perception 評価のためのデータ構造

## 評価単位でのデータ構造

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

## Frame 単位でのデータ構造

- PerceptionFrameResult: 1 pointcloud の入力と、その入力に対しての結果のまとまり
  - 入力データ: 1 pointcloud + List[ground truth objects]
  - object_results (List[DynamicObjectWithPerceptionResult]): Object ごとの評価
  - metrics_score (MetricsScore): Metrics 評価
  - pass_fail_result (PassFailResult): Use case 評価の結果
    - fp_objects (List[DynamicObjectWithPerceptionResult]): Use case 評価で FP (False Positive) の ObjectResult
    - fn_objects (List[DynamicObject]): Use case 評価で FN (False Negative) の DynamicObject
- 詳細は[awml_evaluation/evaluation/result/perception_frame_result.py](../../awml_evaluation/awml_evaluation/evaluation/result/perception_frame_result.py)を参照

```yaml
[2022-06-02 14:08:17,724] [INFO] [perception_lsim.py:286 <module>] Frame result example (frame_results[0]):
{'frame_name': '0',
 'ground_truth_objects': ' --- length of element 98 ---,',
 'metrics_score': {'detection_config': {'center_distance_thresholds': [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
                                        'iou_3d_thresholds': [[], [0.5, 0.5, 0.5, 0.5]],
                                        'iou_bev_thresholds': [[], [0.5, 0.5, 0.5, 0.5]],
                                        'max_x_position_list': [102.4, 102.4, 102.4, 102.4],
                                        'max_y_position_list': [102.4, 102.4, 102.4, 102.4],
                                        'plane_distance_thresholds': [[], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
                                        'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE',
                                                          'AutowareLabel.PEDESTRIAN', 'AutowareLabel.MOTORBIKE']},
                   'maps': ' --- length of element 6 ---,',
                   'prediction_config': {'center_distance_thresholds': [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
                                         'iou_3d_thresholds': [[], [0.5, 0.5, 0.5, 0.5]],
                                         'iou_bev_thresholds': [[], [0.5, 0.5, 0.5, 0.5]],
                                         'max_x_position_list': [102.4, 102.4, 102.4, 102.4],
                                         'max_y_position_list': [102.4, 102.4, 102.4, 102.4],
                                         'plane_distance_thresholds': [[], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
                                         'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE',
                                                           'AutowareLabel.PEDESTRIAN', 'AutowareLabel.MOTORBIKE']},
                   'tracking_config': {'center_distance_thresholds': [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
                                       'iou_3d_thresholds': [[], [0.5, 0.5, 0.5, 0.5]],
                                       'iou_bev_thresholds': [[], [0.5, 0.5, 0.5, 0.5]],
                                       'max_x_position_list': [102.4, 102.4, 102.4, 102.4],
                                       'max_y_position_list': [102.4, 102.4, 102.4, 102.4],
                                       'plane_distance_thresholds': [[], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
                                       'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE',
                                                         'AutowareLabel.PEDESTRIAN', 'AutowareLabel.MOTORBIKE']},
                   'tracking_scores': ' --- length of element 6 ---,'},
 'object_results': ' --- length of element 97 ---,',
 'pass_fail_result': {'critical_ground_truth_objects': ' --- length of element 23 ---,',
                      'critical_object_filter_config': {'max_x_position_list': [30.0, 30.0, 30.0, 30.0],
                                                        'max_y_position_list': [30.0, 30.0, 30.0, 30.0],
                                                        'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE',
                                                                          'AutowareLabel.PEDESTRIAN',
                                                                          'AutowareLabel.MOTORBIKE']},
                      'fn_objects': ' --- length of element 6 ---,',
                      'fp_objects_result': ' --- length of element 15 ---,',
                      'frame_pass_fail_config': {'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE',
                                                                   'AutowareLabel.PEDESTRIAN',
                                                                   'AutowareLabel.MOTORBIKE'],
                                                 'threshold_plane_distance_list': [2.0, 2.0, 2.0, 2.0]},
                      'tp_objects': ' --- length of element 27 ---,'},
 'pointcloud': None,
 'unix_time': 1624164470849887}
```

## Object 単位でのデータ構造

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
[2022-01-11 13:30:22,041] [INFO] [perception_lsim.py:150 <module>] Object result example (frame_results[0].object_results[0]):
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
