# Perception Evaluation Metrics

## [`<class> MetricsScore(...)`](../../../perception_eval/perception_eval/evaluation/metrics/metrics.py)

- A class to evaluate each of detection/tracking/prediction task

| Argument |         type         | Description                               |
| :------- | :------------------: | :---------------------------------------- |
| `config` | `MetricsScoreConfig` | Configuration settings for `MetricsScore` |

- Initialize `detection/tracking/prediction_config` from input MetricsConfig

  - [`detection_config (DetectionMetricsConfig)`](../../../perception_eval/perception_eval/evaluation/metrics/config/detection_metrics_config.py)
  - [`tracking_config (TrackingMetricsConfig)`](../../../perception_eval/perception_eval/evaluation/metrics/config/tracking_metrics_config.py)
  - [`prediction_config (PredictionMetricsConfig)`](../../../perception_eval/perception_eval/evaluation/metrics/config/prediction_metrics_config.py)
  - [`classification_config (ClassificationMetricsConfig)`](../../../perception_eval/perception_eval/evaluation/metrics/config/classification_metrics_config.py)

- Calculate each metrics based on each config

  - 3D evaluation

  | Evaluation Task |      Metrics       |
  | :-------------- | :----------------: |
  | `Detection`     |     mAP / mAPH     |
  | `Tracking`      | mAP / mAPH / CLEAR |
  | `Prediction`    |       [TBD]        |

  - 2D evaluation

  | Task               |   Metrics   |
  | :----------------- | :---------: |
  | `Detection2D`      |     mAP     |
  | `Tracking2D`       | mAP / CLEAR |
  | `Classification2D` |  Accuracy   |

```yaml
[2022-08-09 18:56:45,237] [INFO] [perception_lsim.py:214 <module>] Detection Metrics example (final_metric_score):
{'detection_config': {'center_distance_thresholds': [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
                      'center_distance_bev_thresholds': [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
                      'iou_3d_thresholds': [[0.5, 0.5, 0.5, 0.5]],
                      'iou_bev_thresholds': [[0.5, 0.5, 0.5, 0.5]],
                      'plane_distance_thresholds': [[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
                      'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE', 'AutowareLabel.PEDESTRIAN',
                                        'AutowareLabel.MOTORBIKE']},
 'mean_ap_values': ' --- length of element 6 ---,',
 'num_ground_truth_dict': {<AutowareLabel.PEDESTRIAN: 'pedestrian'>: 7657,
                           <AutowareLabel.MOTORBIKE: 'motorbike'>: 335,
                           <AutowareLabel.BICYCLE: 'bicycle'>: 1319,
                           <AutowareLabel.CAR: 'car'>: 3770},
 'prediction_config': None,
 'prediction_scores': [],
 'tracking_config': None,
 'tracking_scores': []}

 [2022-08-09 18:57:53,888] [INFO] [perception_lsim.py:270 <module>] Tracking Metrics example (tracking_final_metric_score):
{'detection_config': {'center_distance_thresholds': [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
                      'center_distance_bev_thresholds': [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
                      'iou_3d_thresholds': [[0.5, 0.5, 0.5, 0.5]],
                      'iou_bev_thresholds': [[0.5, 0.5, 0.5, 0.5]],
                      'plane_distance_thresholds': [[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
                      'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE', 'AutowareLabel.PEDESTRIAN',
                                        'AutowareLabel.MOTORBIKE']},
 'mean_ap_values': ' --- length of element 6 ---,',
 'num_ground_truth_dict': {<AutowareLabel.PEDESTRIAN: 'pedestrian'>: 7657,
                           <AutowareLabel.MOTORBIKE: 'motorbike'>: 335,
                           <AutowareLabel.BICYCLE: 'bicycle'>: 1319,
                           <AutowareLabel.CAR: 'car'>: 3770},
 'prediction_config': None,
 'prediction_scores': [],
 'tracking_config': {'center_distance_thresholds': [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
                     'iou_3d_thresholds': [[0.5, 0.5, 0.5, 0.5]],
                     'iou_bev_thresholds': [[0.5, 0.5, 0.5, 0.5]],
                     'plane_distance_thresholds': [[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
                     'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE', 'AutowareLabel.PEDESTRIAN',
                                       'AutowareLabel.MOTORBIKE']},
 'tracking_scores': ' --- length of element 6 ---,'}

```

## Detection

### [`<class> Map(...)`](../../../perception_eval/perception_eval/evaluation/metrics/detection/map.py)

- A class to calculate mAP (mean Average Prevision)

  - As internal process, calculating AP (Average Precision) and APH (Average Precision Weighted by Heading)
  - Compute mAP, mAPH by meaning above results for each label

#### AP calculation

- Precision and Recall are formulated based on the decision of TP/FP/FN as below

  <img src="../../fig/perception/precision.png">

  <img src="../../fig/perception/recall.png">

- The area of under curve is AP(APH). The curve is called PR-curve(Precision Recall-curve).

  <img src="../../fig/perception/pr-curve.png" width=480>

- In actual AP(APH) calculation, the curve is interpolated as below.

  <img src="../../fig/perception/pr-curve_interpolate.jpeg" width=480 >

```yaml
[2022-08-09 18:56:45,238] [INFO] [perception_lsim.py:220 <module>] mAP result example (final_metric_score.mean_ap_values[0].aps[0]):
{'aphs': [{'ap': 0.0,
           'fp_list': ' --- length of element 3768 ---,',
           'matching_average': 2.3086792761230375,
           'matching_mode': 'MatchingMode.CENTERDISTANCE',
           'matching_standard_deviation': 1.7793252530202338e-15,
           'matching_threshold_list': [1.0],
           'num_ground_truth': 3770,
           'objects_results_num': 3768,
           'target_labels': ['AutowareLabel.CAR'],
           'tp_list': ' --- length of element 3768 ---,',
           'tp_metrics': {'mode': 'TPMetricsAph'}},
          {'ap': 0.0,
           'fp_list': ' --- length of element 1186 ---,',
           'matching_average': 2.291362716605052,
           'matching_mode': 'MatchingMode.CENTERDISTANCE',
           'matching_standard_deviation': 0.828691650821576,
           'matching_threshold_list': [1.0],
           'num_ground_truth': 1319,
           'objects_results_num': 1186,
           'target_labels': ['AutowareLabel.BICYCLE'],
           'tp_list': ' --- length of element 1186 ---,',
           'tp_metrics': {'mode': 'TPMetricsAph'}},
          {'ap': 0.010345232018551354,
           'fp_list': ' --- length of element 7583 ---,',
           'matching_average': 2.7900883035521864,
           'matching_mode': 'MatchingMode.CENTERDISTANCE',
           'matching_standard_deviation': 3.6498079706351123,
           'matching_threshold_list': [1.0],
           'num_ground_truth': 7657,
           'objects_results_num': 7583,
           'target_labels': ['AutowareLabel.PEDESTRIAN'],
           'tp_list': ' --- length of element 7583 ---,',
           'tp_metrics': {'mode': 'TPMetricsAph'}},
          {'ap': 0.0,
           'fp_list': ' --- length of element 335 ---,',
           'matching_average': 2.308679276123039,
           'matching_mode': 'MatchingMode.CENTERDISTANCE',
           'matching_standard_deviation': 1.8215173398221747e-15,
           'matching_threshold_list': [1.0],
           'num_ground_truth': 335,
           'objects_results_num': 335,
           'target_labels': ['AutowareLabel.MOTORBIKE'],
           'tp_list': ' --- length of element 335 ---,',
           'tp_metrics': {'mode': 'TPMetricsAph'}}],
 'aps': [{'ap': 0.0,
          'fp_list': ' --- length of element 3768 ---,',
          'matching_average': 2.3086792761230375,
          'matching_mode': 'MatchingMode.CENTERDISTANCE',
          'matching_standard_deviation': 1.7793252530202338e-15,
          'matching_threshold_list': [1.0],
          'num_ground_truth': 3770,
          'objects_results_num': 3768,
          'target_labels': ['AutowareLabel.CAR'],
          'tp_list': ' --- length of element 3768 ---,',
          'tp_metrics': {'mode': 'TPMetricsAp'}},
         {'ap': 0.0,
          'fp_list': ' --- length of element 1186 ---,',
          'matching_average': 2.291362716605052,
          'matching_mode': 'MatchingMode.CENTERDISTANCE',
          'matching_standard_deviation': 0.828691650821576,
          'matching_threshold_list': [1.0],
          'num_ground_truth': 1319,
          'objects_results_num': 1186,
          'target_labels': ['AutowareLabel.BICYCLE'],
          'tp_list': ' --- length of element 1186 ---,',
          'tp_metrics': {'mode': 'TPMetricsAp'}},
         {'ap': 0.012201062955606672,
          'fp_list': ' --- length of element 7583 ---,',
          'matching_average': 2.7900883035521864,
          'matching_mode': 'MatchingMode.CENTERDISTANCE',
          'matching_standard_deviation': 3.6498079706351123,
          'matching_threshold_list': [1.0],
          'num_ground_truth': 7657,
          'objects_results_num': 7583,
          'target_labels': ['AutowareLabel.PEDESTRIAN'],
          'tp_list': ' --- length of element 7583 ---,',
          'tp_metrics': {'mode': 'TPMetricsAp'}},
         {'ap': 0.0,
          'fp_list': ' --- length of element 335 ---,',
          'matching_average': 2.308679276123039,
          'matching_mode': 'MatchingMode.CENTERDISTANCE',
          'matching_standard_deviation': 1.8215173398221747e-15,
          'matching_threshold_list': [1.0],
          'num_ground_truth': 335,
          'objects_results_num': 335,
          'target_labels': ['AutowareLabel.MOTORBIKE'],
          'tp_list': ' --- length of element 335 ---,',
          'tp_metrics': {'mode': 'TPMetricsAp'}}],
 'map': 0.003050265738901668,
 'maph': 0.0025863080046378386,
 'matching_mode': 'MatchingMode.CENTERDISTANCE',
 'matching_threshold_list': [1.0, 1.0, 1.0, 1.0],
 'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE', 'AutowareLabel.PEDESTRIAN', 'AutowareLabel.MOTORBIKE']}
```

## Tracking

### `<class> TrackingMetricsScore(...)`

- A class to compute metrics score for tracking.

  - As internal process, calculate CLEAR which includes MOTA (Multi-Object Tracking Accuracy) / MOTP (Multi-Object Tracking Precision) / IDswitch.

- MOTA and MOTP is formulated as below.

  <img src="../../fig/perception/mota.png">

  <img src="../../fig/perception/motp.png" width=240>

```yaml
[2022-08-09 18:57:53,889] [INFO] [perception_lsim.py:282 <module>] CLEAR result example (tracking_final_metric_score.tracking_scores[0].clears[0]):
{'clears': [{'fp': 3759.0,
             'id_switch': 0,
             'mota': 0.0,
             'motp': 0.8297818944410861,
             'objects_results_num': 3771,
             'tp': 12.0,
             'tp_matching_score': 9.957382733293032},
            {'fp': 1189.0,
             'id_switch': 0,
             'mota': 0.0,
             'motp': inf,
             'objects_results_num': 1189,
             'tp': 0.0,
             'tp_matching_score': 0.0},
            {'fp': 6834.0,
             'id_switch': 11,
             'mota': 0.0,
             'motp': 0.6495895085858145,
             'objects_results_num': 7526,
             'tp': 692.0,
             'tp_matching_score': 449.5159399413837},
            {'fp': 335.0,
             'id_switch': 0,
             'mota': 0.0,
             'motp': inf,
             'objects_results_num': 335,
             'tp': 0.0,
             'tp_matching_score': 0.0}],
 'matching_mode': 'MatchingMode.CENTERDISTANCE',
 'target_labels': ['AutowareLabel.CAR', 'AutowareLabel.BICYCLE', 'AutowareLabel.PEDESTRIAN', 'AutowareLabel.MOTORBIKE']}
```

### ID switch

<img src="../../fig/perception/ID_switch_design.svg">

## Matching

- A Class to represent the method of matching estimation and GT
  - For the details, see [perception_eval/evaluation/matching/object_matching.py](../../../perception_eval/perception_eval/evaluation/matching/object_matching.py)

| Matching Method    | Value                                                                 |
| ------------------ | --------------------------------------------------------------------- |
| Center Distance 3D | Center distance between two objects                                   |
| IoU 2D             | IoU 2D score between two objects (In 3D evaluation, BEV is used)      |
| IoU 3D             | IoU 3D score between two objects                                      |
| Plane Distance     | Nearest plane distance between two objects(explain the details below) |

- We match estimated objects and GTs by the following policy:

1. Matching the nearest estimated objects and GTs which have the same label primary.
2. Matching the nearest estimated objects and GTs regardless of their label.

<img src="../../fig/perception/object_matching1.svg">

- In case of specifying uuid, the pair will be generated as following process.

1. Filter GTs which do not have specified uuid.
2. Matching objects.
3. Filter estimations which do not have matching pair.

<img src="../../fig/perception/object_matching2.svg">

### Plane distance

- In usecase evaluation, determine TP/FP depends on **RMS of two points being the nearest from ego** between GT and estimation.

  1. Choose two points being the nearest from ego.
  2. From the two pairs of end points between the faces, the pair with the shortest total distance is selected and set as two points near the vehicle.
  3. Take the root mean square of the distance of each pair and call it \*RMS of the distance between two points near the vehicle\*\*.

- Example
  1. For GT, choose plane g3g4 near from ego. For estimation, choose plane d3d4.
  2. In this case, there are two types of end points pair, (g3d3, g4d4) and (g3d4, g4d3). Choose the pair with the shorted total distance, (g3d3, g4d4), in this case.
  3. Then RMS = sqrt ( ( g3d3^2 + g4d4^2 ) / 2 )
  - For more information, see `<func> get_uc_plane_distance()`.
  - Background of 1: Because it can not identify object's depth, the most confident near point from ego is selected.

![pipeline](../../fig/perception/uc_plane_distance.svg)

## TP Metrics

- A class to return TP value
  - For the details，see [perception_eval/evaluation/metrics/detection/tp_metrics.py](../../../perception_eval/perception_eval/evaluation/metrics/detection/tp_metrics.py)

| TP Metrics          | Value                                      |
| ------------------- | ------------------------------------------ |
| TPMetricsAp         | 1.0                                        |
| TPMetricsAph        | Heading error between two objects[-pi, pi] |
| TPMetricsConfidence | Confidence of estimation                   |

## Advanced detection metrics (driving-aware)

Opt-in metrics ported from [`tier4/autoware-ml` PR #109](https://github.com/tier4/autoware-ml/pull/109)
(commit `fcf86419`). They run in addition to mAP/mAPH and never change the existing values. Enable them
by adding one nested section to the detection `evaluation_config_dict`; when the section is absent the
evaluator behaves exactly as before.

```yaml
advanced_detection_metrics:
  ranges: # optional radial BEV windows [min, max) on the box center
    - { name: 0_30, min_distance: 0.0, max_distance: 30.0 }
    - { name: 30_60, min_distance: 30.0, max_distance: 60.0 }
  class_groups: # optional full partition of target_labels (AutowareLabel names)
    grouped_vehicle: [car, truck, bus]
    grouped_vru: [pedestrian, bicycle, motorbike]
    grouped_static: [hazard, unknown]
  filters: # optional spatial views (the whole-scene view is always evaluated)
    - { name: corridor, type: corridor, width_m: 3.0 }
    - { name: road, type: region, regions: [road, road_shoulder, crosswalk] }
    - { name: collision, type: collision }
  components: # at least one
    - { type: corner_error, tp_threshold: 2.0, percentiles: [95.0] }
    - { type: heading_flip, tp_threshold: 2.0, flip_threshold: 1.57079632679 }
    - { type: nearest_surface_error, tp_threshold: 2.0 }
    - { type: calibration, tp_threshold: 2.0, num_bins: 15 }
    - { type: confident_error, tp_threshold: 2.0, min_score: 0.1, score_threshold: 0.5 }
    - { type: confusion_matrix, match_threshold: 2.0, min_score: 0.1 }
    - { type: critical_fp_fn, confidences: [0.3, 0.5], match_threshold: 2.0 }
    - { type: collision_weighted_map, thresholds: [0.5, 1.0, 2.0, 4.0], decay: 0.5 }
  map: # required by region/collision filters and by critical_fp_fn / collision_weighted_map
    resolver: t4_scene_directory # <scene dir>/map/lanelet2_map.osm (or <scene dir>/*/map/...)
    # resolver: explicit
    # mapping: { "/path/to/scene": "/path/to/lanelet2_map.osm" }
```

`type` tokens form a closed registry; unknown tokens or keys are rejected at configuration time.

### Output

`MetricsScore.detection_metric_report` (`MetricReport`) holds:

- `values`: `{key: float}` with slash-separated keys
  `detection/<taxonomy?>/<filter?>/<range?>/<metric-key>`, e.g. `detection/corner_mean_car`,
  `detection/road/0m_30m/corner_p95_car`, `detection/grouped/corridor/mflip_rate`. The `grouped`
  level appears only when `class_groups` is configured, the filter level only for non-identity
  filters, the range level only when `ranges` is configured. `MetricReport.to_flat_keys()` gives
  `detection_road_0m_30m_corner_p95_car` for consumers that cannot accept slashes.
- `coverage`: `{view: (covered_frames, seen_frames)}` for every filter and, when TTC is used, `ttc`.
- `warnings`: partial coverage, skipped frames (e.g. polygon-shaped objects) and similar notes.

`NaN` means "undefined" (no true positive for a class, zero covered frames, ...). It is never
reported as `0`.

### Matching

Components use their own score-ordered greedy matcher (BEV center distance, per frame; ties go to
the lowest ground-truth index). Boxes are represented in `base_link` as
`[cx, cy, cz, dx=length, dy=width, dz=height, yaw]`; objects in the `map` frame are transformed
with the frame's `base_link -> map` transform. Polygon-shaped objects are not supported and skip the
frame with a warning.

### Components

| `type`                   | Definition (defaults)                                                                                                                                                                                                                    | Keys                                                                                                                    |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `corner_error`           | Mean BEV corner displacement of each TP (`tp_threshold=2.0`) under the best cyclic corner assignment; mean / max / `percentiles=[95]` per class, macro mean over classes.                                                                | `corner_mean_<c>`, `corner_max_<c>`, `corner_p95_<c>`, `mcorner_mean`, `mcorner_max`                                    |
| `heading_flip`           | Fraction of TPs whose absolute wrapped yaw error exceeds `flip_threshold=pi/2`.                                                                                                                                                          | `flip_rate_<c>`, `flip_count_<c>`, `mflip_rate`                                                                         |
| `nearest_surface_error`  | Signed `d(pred) - d(gt)` where `d` is the distance from ego to the nearest point of the BEV footprint; positive = predicted near face too far (late braking). Mean, `low_percentile=5`, `high_percentile=95`, absolute max.              | `nsurf_mean_<c>`, `nsurf_low_<c>`, `nsurf_high_<c>`, `nsurf_absmax_<c>`, `mnsurf_high`, `mnsurf_absmax`                 |
| `calibration`            | Expected calibration error of the score against TP precision over `num_bins=15` equal-width bins; scores must be probabilities in `[0, 1]`.                                                                                              | `ece` (pooled), `ece_macro` (mean over predicted classes)                                                               |
| `confident_error`        | Among FPs with score `>= min_score=0.1`, the fraction with score `>= score_threshold=0.5`.                                                                                                                                               | `confident_error_rate`, `confident_error_count`, `confident_errors_per_frame`                                           |
| `confusion_matrix`       | Class-agnostic greedy match at `match_threshold=2.0` of predictions with score `>= min_score=0.1`; counts matched `(true, predicted)` label pairs only.                                                                                  | `confusion_<true>__<pred>`                                                                                              |
| `critical_fp_fn`         | At each confidence in `confidences=[0.3, 0.5]`, class-agnostic match at `match_threshold=2.0`; count unmatched predictions / GTs whose reachability TTC is finite, divided by the number of TTC-covered frames. Needs `map`.             | `critical_fp_<conf>`, `critical_fn_<conf>`, `critical_fp_<c>_<conf>`, `critical_fn_<c>_<conf>` (conf token e.g. `0p5m`) |
| `collision_weighted_map` | nuScenes-style AP (same convention as `Ap`) where every object is weighted by `exp(-decay * TTC)` (`decay=0.5`, unreachable = 0); a TP inherits its GT weight, an FP keeps its own; averaged over `thresholds` and classes. Needs `map`. | `cw_mAP`, `cw_mAP_<c>`                                                                                                  |

### Filters

| `type`      | Elements kept                                                                                                                                                                                                  | Needs map |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| `corridor`  | Boxes whose forward (`x >= 0`) footprint overlaps the strip `abs(y) <= width_m / 2` in `base_link`.                                                                                                            | no        |
| `region`    | Boxes whose footprint intersects the union of the lanelet2 `regions` (lanelet `subtype` or area way `type`); `margin_m` erodes the mapped surface inward or, with `expand: true`, dilates the region outward.  | yes       |
| `collision` | Boxes intersecting the ego reachable region within the horizon (bounded steering at the lanelet speed limit, clipped to the drivable lanelets). Parameters: `horizon_s`, `dt_s`, `max_lateral_accel_mps2`, ... | yes       |

Frames whose scene has no lanelet map (or no ego pose) are excluded only from the map-dependent
views; the whole-scene view keeps them. A view with zero covered frames reports `NaN` for every key
and one warning.

### Reachability model (TTC)

Every agent, including ego, moves at the maximum legal speed of its class: wheeled classes
(`car`, `truck`, `bus`, `motorbike`) follow constant-curvature arcs at the lanelet `speed_limit`
(off-map fallback `map.max_speed_mps=16.7`), VRUs (`pedestrian` 3 m/s, `bicycle` 6 m/s, `animal`
4 m/s) move isotropically, static classes (`hazard`, `unknown`) keep their footprint. TTC is the
first `t` at which the reachable-at-`t` sets of ego and the object overlap, or `inf`. A lead vehicle
travelling at the same speed is therefore unreachable and never critical. Override the class table
with `map.collision_kinds` / `map.vru_speeds`.

### Example

```python
score = evaluator.get_scene_result()
report = score.detection_metric_report  # None when the section is not configured
print(report.summary())
value = report.values["detection/road/0m_30m/corner_p95_car"]
flat = report.to_flat_keys()
```
