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

### PerceptionFrameResult

- 入力データ: 1 pointcloud + List[ground truth objects]
- object_results (List[DynamicObjectWithPerceptionResult]): Object ごとの評価
- metrics_score (MetricsScore): Metrics 評価
- pass_fail_result (PassFailResult): Use case 評価の結果
  - fp_objects (List[DynamicObjectWithPerceptionResult]): Use case 評価で FP (False Positive) の ObjectResult
  - fn_objects (List[DynamicObject]): Use case 評価で FN (False Negative) の DynamicObject
- 詳細は[awml_evaluation/evaluation/result/perception_frame_result.py](../../awml_evaluation/awml_evaluation/evaluation/result/perception_frame_result.py)を参照

```yaml
[2022-08-10 10:38:11,341] [INFO] [perception_lsim.py:258 <module>] Frame result example (frame_results[0]):
{'frame_ground_truth': {'ego2map': ' --- length of element 4 ---,',
                        'frame_id': 'map',
                        'frame_name': '0',
                        'objects': ' --- length of element 89 ---,',
                        'pointcloud': None,
                        'unix_time': 1624164470849887},
 'frame_id': 'map',
 'frame_name': '0',
 'metrics_score': {'detection_config': {'center_distance_thresholds': ' --- length of element 2 ---,',
                                        'iou_3d_thresholds': [' --- length of element 4 ---,'],
                                        'iou_bev_thresholds': [' --- length of element 4 ---,'],
                                        'plane_distance_thresholds': ' --- length of element 2 ---,',
                                        'target_labels': ' --- length of element 4 ---,'},
                   'maps': ' --- length of element 6 ---,',
                   'prediction_config': None,
                   'prediction_scores': [],
                   'tracking_config': {'center_distance_thresholds': ' --- length of element 2 ---,',
                                       'iou_3d_thresholds': [' --- length of element 4 ---,'],
                                       'iou_bev_thresholds': [' --- length of element 4 ---,'],
                                       'plane_distance_thresholds': ' --- length of element 2 ---,',
                                       'target_labels': ' --- length of element 4 ---,'},
                   'tracking_scores': ' --- length of element 6 ---,'},
 'object_results': ' --- length of element 88 ---,',
 'pass_fail_result': {'critical_ground_truth_objects': ' --- length of element 23 ---,',
                      'critical_object_filter_config': {'confidence_threshold_list': None,
                                                        'filtering_params': {'confidence_threshold_list': None,
                                                                             'max_distance_list': None,
                                                                             'max_x_position_list': ' --- length of '
                                                                                                    'element 4 ---,',
                                                                             'max_y_position_list': ' --- length of '
                                                                                                    'element 4 ---,',
                                                                             'min_distance_list': None,
                                                                             'min_point_numbers': None,
                                                                             'target_labels': ' --- length of element '
                                                                                              '4 ---,',
                                                                             'target_uuids': None},
                                                        'max_distance_list': None,
                                                        'max_x_position_list': ' --- length of element 4 ---,',
                                                        'max_y_position_list': ' --- length of element 4 ---,',
                                                        'min_distance_list': None,
                                                        'min_point_numbers': None,
                                                        'target_labels': ' --- length of element 4 ---,',
                                                        'target_uuids': None},
                      'ego2map': ' --- length of element 4 ---,',
                      'fn_objects': ' --- length of element 17 ---,',
                      'fp_objects_result': ' --- length of element 17 ---,',
                      'frame_id': 'map',
                      'frame_pass_fail_config': {'confidence_threshold_list': None,
                                                 'plane_distance_threshold_list': ' --- length of element 4 ---,',
                                                 'target_labels': ' --- length of element 4 ---,'},
                      'tp_objects': ' --- length of element 22 ---,'},
 'target_labels': ' --- length of element 4 ---,',
 'unix_time': 1624164470849887}
```

## Object 単位でのデータ構造

### DynamicObjectWithPerceptionResult

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
[2022-08-09 18:56:45,237] [INFO] [perception_lsim.py:208 <module>] Object result example (frame_results[0].object_results[0]):
{'center_distance': {'mode': 'MatchingMode.CENTERDISTANCE', 'value': 0.5051040904718623},
 'estimated_object': {'pointcloud_num': 113,
                      'predicted_confidence': None,
                      'predicted_path': None,
                      'semantic_label': 'AutowareLabel.PEDESTRIAN',
                      'semantic_score': 0.6965261662052492,
                      'state': {'orientation': {'q': [0.9394992040041219, 0.016502364866547787, -0.0022243908564357263,
                                                      0.34214612333731087]},
                                'position': [12.830938811123243, -28.461605817964994, 0.4005466639700647],
                                'size': [0.758, 1.138, 1.861],
                                'velocity': [-0.8304898624100426, 1.0906433133424918, -0.0700413136982921]},
                      'tracked_path': None,
                      'unix_time': 1624164470849887,
                      'uuid': 'c28556c19064ad491ff1dc438a38a3a7'},
 'ground_truth_object': {'pointcloud_num': 128,
                         'predicted_confidence': None,
                         'predicted_path': None,
                         'semantic_label': 'AutowareLabel.PEDESTRIAN',
                         'semantic_score': 1.0,
                         'state': {'orientation': {'q': [0.9947797942640241, 0.012007913693226124,
                                                         0.0001662701515344235, 0.10133579469762255]},
                                   'position': [12.629186624221628, -28.881230126682482, 0.2047364578387355],
                                   'size': [0.822, 1.199, 1.822],
                                   'velocity': [-0.570336411550396, 1.520897097467723, -0.48028329393123725]},
                         'tracked_path': None,
                         'unix_time': 1624164470849887,
                         'uuid': '912ae043cbc5a6ad4950f5ac0e94778e'},
 'iou_3d': {'mode': 'MatchingMode.IOU3D', 'value': 0.24986054835978477},
 'iou_bev': {'mode': 'MatchingMode.IOUBEV', 'value': 0.2878950915821158},
 'is_label_correct': True,
 'plane_distance': {'estimated_nn_plane': [[13.02303048243653, -27.805782945059786, 0.4205253823079967],
                                           [12.151479338961119, -28.537310518275785, 0.40291816982528683]],
                    'ground_truth_nn_plane': [[13.133512578820893, -28.35791997396456, 0.21582995052066864],
                                              [11.959137571117656, -28.59965947050987, 0.213308623084141]],
                    'mode': 'MatchingMode.PLANEDISTANCE',
                    'value': 0.4230510251796533}}
```
