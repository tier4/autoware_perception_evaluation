# Point-Cloud Semantic Segmentation Evaluation

`perception_eval` evaluates per-point semantic segmentation through a dedicated, streaming manager.
The metrics are a port of `tier4/autoware-ml` (commit `fcf86419`, PR #109) into a framework-free
NumPy/SciPy implementation. Frames are supplied as already-decoded arrays, so any model or dataset
adapter can feed the evaluator.

## Input contract

```python
from perception_eval.evaluation.metrics.segmentation import SegmentationFrame

frame = SegmentationFrame(
    frame_name="0",
    scene_id="db_v1/scene_uuid/0",     # resolves a lanelet map; None when unknown
    coordinates=xyz,                    # (N, 3+) float, base_link
    targets=target_labels,              # (N,) int, `ignore_index` allowed
    predictions=predicted_labels,       # (N,) int
    probabilities=probabilities,        # (N, C) float, rows sum to one
    transforms=frame_ground_truth.transforms,  # TransformDict with (BASE_LINK, MAP) for map filters
    gt_objects=tuple(frame_ground_truth.objects),  # detection boxes for the partial-detection metric
)
```

Validation (raises `ValueError`):

| Check                                        | Note                                                                                                                                   |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| shapes and integer label dtypes              | structural, checked at construction                                                                                                    |
| `probabilities.shape[1] == len(class_names)` | class-column count must equal the configuration                                                                                        |
| values in `[0, 1]` and finite                | `probability_tolerance` (default `1e-6`) allows float noise                                                                            |
| rows sum to one                              | `probability_sum_tolerance` (default `1e-3`)                                                                                           |
| `predictions == argmax(probabilities)`       | `check_argmax: true` by default; disable if the reported class is chosen otherwise, the gathered probability then stays the confidence |
| `gt_objects` are `BOUNDING_BOX`              | only when `partial_detection` is configured                                                                                            |

Derived per point: `confidence = probabilities[prediction]` and normalized entropy
`-sum(p log p) / log(C)` (zero terms treated as zero). A point is _valid_ when
`target != ignore_index`, `0 <= target < C` and `0 <= prediction < C`; invalid points are excluded
from every metric.

## Usage

```python
from perception_eval.config import SegmentationEvaluationConfig
from perception_eval.manager import SegmentationEvaluationManager

config = SegmentationEvaluationConfig(
    dataset_paths=[],                # arrays are supplied externally
    frame_id="base_link",
    result_root_directory="data/result/{TIME}/",
    evaluation_config_dict={
        "evaluation_task": "segmentation",
        "class_names": ["car", "truck", "pedestrian", "road", "sidewalk", "vegetation"],
        "ignore_index": -1,
        "ranges": [{"name": "0_30", "min_distance": 0.0, "max_distance": 30.0}],
        "class_groups": {"vehicle": ["car", "truck"], "vru": ["pedestrian"], "flat": ["road", "sidewalk"], "other": ["vegetation"]},
        "filters": [
            {"name": "corridor", "type": "corridor", "width_m": 3.0},
            {"name": "region_road", "type": "region", "regions": ["road", "road_shoulder"]},
        ],
        "components": [
            {"type": "confusion_matrix"}, {"type": "iou"}, {"type": "accuracy"}, {"type": "precision_recall_f1"},
            {"type": "calibration", "num_bins": 15},
            {"type": "uncertainty_usefulness", "num_bins": 8192},
            {"type": "confident_error", "entropy_threshold": 0.3},
            {"type": "error_clusters", "cluster_radius": 0.5, "min_cluster_points": 1},
            {"type": "tolerant_error", "radius": 0.2},
            {"type": "partial_detection", "half_saturation": 1.0, "min_points": 1},
        ],
        "box_label_to_seg_class": {"car": "car", "truck": "truck", "pedestrian": "pedestrian"},
        "map": {"resolver": "t4_scene_directory", "data_root": "/path/to/t4"},
        "check_argmax": True,
    },
)
manager = SegmentationEvaluationManager(config)
for frame in frames:                 # SegmentationFrame instances, one at a time
    summary = manager.add_frame(frame)
report = manager.get_scene_result(save_report=True)   # writes <log_directory>/segmentation_metrics.json
print(report)
```

`manager.frame_from_ground_truth(ground_truth, targets, predictions, probabilities)` builds a frame
from a loaded `FrameGroundTruth` (coordinates default to the `LIDAR_CONCAT`/`LIDAR_TOP` point
cloud, so `load_raw_data=True` and `load_ground_truth=True` are needed for that path).

The mock `python -m test.segmentation_lsim --use_tmpdir` runs the whole pipeline on synthetic frames.

## Configuration keys

| Key                         | Default  | Meaning                                                                        |
| --------------------------- | -------- | ------------------------------------------------------------------------------ |
| `class_names`               | required | trained classes in probability-column order                                    |
| `ignore_index`              | `-1`     | target label excluded everywhere                                               |
| `ranges`                    | `[]`     | radial BEV windows `[min, max)` on point / box-center distance                 |
| `class_groups`              | `null`   | full partition of `class_names`; adds the `grouped` taxonomy view              |
| `filters`                   | `[]`     | `corridor` (map-free), `region`, `collision` (need `map`)                      |
| `components`                | `[]`     | closed registry, see below                                                     |
| `map`                       | `null`   | `{resolver: t4_scene_directory, data_root}` or `{resolver: explicit, mapping}` |
| `box_label_to_seg_class`    | `{}`     | `AutowareLabel` value -> segmentation class of the points inside such a box    |
| `check_argmax`              | `true`   | require `predictions == argmax(probabilities)`                                 |
| `probability_tolerance`     | `1e-6`   | allowed excursion outside `[0, 1]`                                             |
| `probability_sum_tolerance` | `1e-3`   | allowed deviation of each row sum from one                                     |
| `include_confusion_cells`   | `true`   | emit the `confusion_<true>__<pred>` keys                                       |

Unknown keys, unknown component types or parameters, duplicate names, non-partitioning class
groups and unmapped labels all fail at configuration time (`MetricsConfigError`).

## Metric keys

Every value is emitted under a slash-separated key:

```text
segmentation/<taxonomy?>/<filter?>/<range?>/<metric-key>
```

The taxonomy level is present only for the grouped view (`grouped`), the filter level only for
non-identity filters, the range level (`0m_30m`, `30m_inf`, ...) only when ranges are configured.
Examples: `segmentation/mIoU`, `segmentation/0m_30m/error_rate`,
`segmentation/grouped/region_road/error_clusters_per_frame`. `report.to_flat_keys()` gives the
legacy flat form (`segmentation_grouped_region_road_error_clusters_per_frame`).

| Component (`type`)            | Parameters (defaults)                        | Keys                                                                                                                                                                                  |
| ----------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `confusion_matrix`            | -                                            | `confusion_<true>__<pred>` raw point counts                                                                                                                                           |
| `iou`                         | -                                            | `mIoU` (macro over classes with GT support), `fwIoU`, `iou_<c>` for supported classes                                                                                                 |
| `accuracy`                    | -                                            | `acc` (correct / valid points)                                                                                                                                                        |
| `precision_recall_f1`         | -                                            | `mRecall`, `mPrecision`, `mF1`, `recall_<c>`, `precision_<c>`, `f1_<c>`                                                                                                               |
| `calibration` (E1)            | `num_bins=15`                                | `ece` (all points), `ece_macro` (mean of per-_predicted_-class ECE)                                                                                                                   |
| `uncertainty_usefulness` (E2) | `num_bins=8192`                              | `entropy_auroc` (tie-aware AUROC of entropy as an error detector from two histograms), `mean_entropy_wrong`, `mean_entropy_correct`                                                   |
| `confident_error` (E3)        | `entropy_threshold=0.3`                      | `confident_error_rate` = wrong points with entropy `< threshold` / wrong points, `confident_error_count`                                                                              |
| `error_clusters` (D1)         | `cluster_radius=0.5`, `min_cluster_points=1` | `error_rate`, `error_cluster_count`, `error_clusters_per_frame`, `error_rate_<c>`, `error_cluster_count_<c>` (per true class)                                                         |
| `tolerant_error` (D3)         | `radius=0.2`                                 | `tolerant_error_rate`, `tolerant_error_count`, per class; a wrong point is rescued when any point within `radius` predicts its true class; `radius=0` equals the strict error rate    |
| `partial_detection` (D2)      | `half_saturation=1.0`, `min_points=1`        | `pd_score_<det class>`, `mpd_score`, `pd_skipped_low_point_boxes`; credit `(k/(k+h)) / (n/(n+h))` for `k` correct of `n` points inside the yaw-aware BEV footprint; raw taxonomy only |

NaN rules: a rate whose denominator is zero is `NaN` (never `0`); counts are `0.0`. Per-class
IoU/precision/recall keys are emitted only for classes with ground-truth support. In the grouped
view probabilities are folded _before_ deriving confidence and entropy, so the grouped confidence is
the summed probability mass of the predicted group.

## Coverage of map-dependent filters

Region and collision filters need an ego pose (`(BASE_LINK, MAP)` transform) and a lanelet map for
the frame's `scene_id`. Frames without either are excluded from that filter's views only; the report
carries `coverage[filter] = (covered_frames, seen_frames)` and one warning per partially covered
filter. A filter that covered no frame reports `NaN` for every key, never a misleading zero.

## Memory guarantee

The manager retains no frame: confusion matrices are fixed-size `(F+1, R+1, C, C)` integer arrays,
calibration and entropy statistics are fixed-size bins, and cluster / neighbourhood metrics are
computed per frame and reduced to counters. Peak memory is therefore bounded by the largest single
frame, not by the scene length (see `test_manager.py::test_memory_stays_bounded`).
