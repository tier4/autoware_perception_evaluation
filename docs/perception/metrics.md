# Perception Evaluation Metrics

## MetricsScore

- detection/tracking/prediction の各評価指標を実行する class

- config (MetricsScoreConfig)
  - detection_config (DetectionMetricsConfig)
  - tracking_config (TrackingMetricsConfig)
  - prediction_config (PredictionMetricsConfig)

```yaml
[2022-05-26 14:17:27,687] [INFO] [lsim.py:297 <module>] Metrics example (final_metric_score):
{'detection_config': {'center_distance_thresholds': [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
                      'iou_3d_thresholds': [[0.5, 0.5, 0.5, 0.5]],
                      'iou_bev_thresholds': [[0.5, 0.5, 0.5, 0.5]],
                      'max_x_position_list': [102.4, 102.4, 102.4, 102.4],
                      'max_y_position_list': [102.4, 102.4, 102.4, 102.4],
                      'plane_distance_thresholds': [[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
                      'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE', 'AutowareLabel.PEDESTRIAN',
                                        'AutowareLabel.MOTORBIKE']},
 'maps': ' --- length of element 6 ---,',
 'prediction_config': {'center_distance_thresholds': [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
                       'iou_3d_thresholds': [[0.5, 0.5, 0.5, 0.5]],
                       'iou_bev_thresholds': [[0.5, 0.5, 0.5, 0.5]],
                       'max_x_position_list': [102.4, 102.4, 102.4, 102.4],
                       'max_y_position_list': [102.4, 102.4, 102.4, 102.4],
                       'plane_distance_thresholds': [[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
                       'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE', 'AutowareLabel.PEDESTRIAN',
                                         'AutowareLabel.MOTORBIKE']},
 'tracking_config': {'center_distance_thresholds': [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
                     'iou_3d_thresholds': [[0.5, 0.5, 0.5, 0.5]],
                     'iou_bev_thresholds': [[0.5, 0.5, 0.5, 0.5]],
                     'max_x_position_list': [102.4, 102.4, 102.4, 102.4],
                     'max_y_position_list': [102.4, 102.4, 102.4, 102.4],
                     'plane_distance_thresholds': [[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
                     'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE', 'AutowareLabel.PEDESTRIAN',
                                       'AutowareLabel.MOTORBIKE']},
 'tracking_scores': ' --- length of element 6 ---,'}
```

### Detection

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

### Tracking

Tracking 評価では，以下に示す CLEAR metrics に加えて，上記の mAP も計算される．

- TrackingMetricsScore: tracking の score を計算した class
  - clears (List[CLEAR]): 各 label の CLEAR

```yaml
{
  "clears":
    [
      {
        "fp": 3768.0,
        "frame_num": 200,
        "ground_truth_objects_num": 3770,
        "id_switch": 0,
        "matching_mode_": "MatchingMode.CENTERDISTANCE",
        "matching_threshold_list_": [1.0],
        "max_x_position_list_": [102.4],
        "max_y_position_list_": [102.4],
        "metrics_filed_": ["MOTA", "MOTP"],
        "mota": 0.0,
        "motp": inf,
        "objects_results_num": 3768,
        "target_labels_": ["AutowareLabel.CAR"],
        "tp": 0.0,
        "tp_matching_score": 0.0,
        "tp_metrics_": { "mode": "TPMetricsAp" },
      },
      {
        "fp": 1186.0,
        "frame_num": 200,
        "ground_truth_objects_num": 1319,
        "id_switch": 0,
        "matching_mode_": "MatchingMode.CENTERDISTANCE",
        "matching_threshold_list_": [1.0],
        "max_x_position_list_": [102.4],
        "max_y_position_list_": [102.4],
        "metrics_filed_": ["MOTA", "MOTP"],
        "mota": 0.0,
        "motp": inf,
        "objects_results_num": 1186,
        "target_labels_": ["AutowareLabel.BICYCLE"],
        "tp": 0.0,
        "tp_matching_score": 0.0,
        "tp_metrics_": { "mode": "TPMetricsAp" },
      },
      {
        "fp": 6809.0,
        "frame_num": 200,
        "ground_truth_objects_num": 7657,
        "id_switch": 6,
        "matching_mode_": "MatchingMode.CENTERDISTANCE",
        "matching_threshold_list_": [1.0],
        "max_x_position_list_": [102.4],
        "max_y_position_list_": [102.4],
        "metrics_filed_": ["MOTA", "MOTP"],
        "mota": 0.0,
        "motp": 0.7065867725109557,
        "objects_results_num": 7583,
        "target_labels_": ["AutowareLabel.PEDESTRIAN"],
        "tp": 774.0,
        "tp_matching_score": 546.8981619234797,
        "tp_metrics_": { "mode": "TPMetricsAp" },
      },
      {
        "fp": 335.0,
        "frame_num": 200,
        "ground_truth_objects_num": 335,
        "id_switch": 0,
        "matching_mode_": "MatchingMode.CENTERDISTANCE",
        "matching_threshold_list_": [1.0],
        "max_x_position_list_": [102.4],
        "max_y_position_list_": [102.4],
        "metrics_filed_": ["MOTA", "MOTP"],
        "mota": 0.0,
        "motp": inf,
        "objects_results_num": 335,
        "target_labels_": ["AutowareLabel.MOTORBIKE"],
        "tp": 0.0,
        "tp_matching_score": 0.0,
        "tp_metrics_": { "mode": "TPMetricsAp" },
      },
    ],
  "frame_num": 200,
  "matching_mode": "MatchingMode.CENTERDISTANCE",
}
```

## Matching

- 予測 object と Ground Truth のマッチング方式の class
  - 詳細は，[awml_evaluation/evaluation/matching/object_matching.py](../../awml_evaluation/awml_evaluation/evaluation/matching/object_matching.py)を参照

| Matching Method    | Value                                             |
| ------------------ | ------------------------------------------------- |
| Center Distance 3D | 2 つの object の 3D 中心間距離                    |
| IoU BEV            | 2 つの object のの IoU BEV の値                   |
| IoU 3D             | 2 つの object の 3D IoU の値                      |
| Plane Distance     | 2 つの object の近傍 2 点の距離の RMS(詳細は後述) |

### Plane distance

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

## TP Metrics

- True Positive 時の値を返す class
  - 詳細は，[awml_evaluation/evaluation/metrics/detection/tp_metrics.py](../../awml_evaluation/awml_evaluation/evaluation/metrics/detection/tp_metrics.py)を参照

| TP Metrics          | Value                         |
| ------------------- | ----------------------------- |
| TPMetricsAp         | 1.0                           |
| TPMetricsAph        | 2 つの object の heading 残差 |
| TPMetricsConfidence | 予測 object の confidence     |
