# Implementation Plan: Driving-Aware Detection and Point-Cloud Segmentation Metrics

## 1. Purpose

This document describes how to port the metrics and supporting evaluation axes introduced by
[`tier4/autoware-ml` PR #109](https://github.com/tier4/autoware-ml/pull/109), specifically the
commit range
[`496943b3..fcf86419`](https://github.com/tier4/autoware-ml/compare/496943b31a2e1ccf97e29e7221982215c1b59bfb...fcf864196ad6460438980b236d4a1c24ffab1e14),
into `perception_eval`.

The target is behavioral parity with the source implementation while preserving
`perception_eval`'s existing public configuration and mAP/mAPH results. The implementation should
be usable for both offline scene evaluation and future model-evaluation adapters, without taking
a runtime dependency on PyTorch, TorchMetrics, Hydra, or `autoware_ml`.

## 2. Scope

The source change contains more than individual metric formulas. It adds three reusable axes that
must be implemented before all metrics can be reproduced:

1. **Taxonomy axis**: evaluate trained classes and behavior-oriented class groups side by side.
2. **Spatial axis**: evaluate the whole scene, radial ranges, lanelet regions, a forward corridor,
   and an ego-reachable collision region.
3. **Criticality axis**: calculate reachability-based time-to-collision (TTC), then use TTC either
   as a binary critical-set selector or a continuous object weight.

The implementation scope is therefore:

- eight detection metrics;
- seven point-cloud segmentation metrics;
- class grouping, range slicing, and spatial filtering;
- Lanelet2 OSM parsing and per-scene map resolution;
- the reachability/TTC model used by the two criticality metrics;
- stable metric-key generation, coverage reporting, configuration, serialization, tests, and
  documentation;
- a new point-cloud segmentation evaluation input contract, because `perception_eval` currently
  has no semantic-segmentation evaluation task.

Existing detection mAP/mAPH remain the default. The new metrics are opt-in until parity and
performance have been validated on representative T4 datasets.

## 3. Source Change Summary

The compared range contains nine commits:

| Area                                  | Source commits                     |
| ------------------------------------- | ---------------------------------- |
| Filter and taxonomy axes              | `d18ebba9`, `6ad71a1e`, `009f57a6` |
| Reachability and criticality          | `8420e6e8`, `bac1f618`             |
| Segmentation state and metrics        | `0d34acb2`, `488b452a`, `a0464217` |
| Model-to-metric segmentation contract | `fcf86419`                         |

The source implementation is a TorchMetrics state engine for training-time distributed
evaluation. `perception_eval` already owns fully materialized per-frame and per-scene results, so
the formulas and contracts should be ported, but TorchMetrics state registration, DDP gathering,
stage binding, and Hydra object injection should not be copied.

## 4. Metric Contracts

### 4.1 Detection metrics

All box-based formulas operate on 3D boxes represented in `base_link` as
`[cx, cy, cz, dx, dy, dz, yaw, ...]`. Unless stated otherwise, matching is score-ordered greedy
BEV center-distance matching, performed independently in each frame. The source defaults use a
2.0 m true-positive threshold.

| Metric                                  | Definition and defaults                                                                                                                                                                                                                                                                                      | Required input                                         | Required outputs                                                                         |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| **Corner displacement error (A1)**      | For a matched pair, calculate the four BEV corners and use the mean corner displacement under the best of the four cyclic corner assignments. Aggregate true positives at `tp_threshold=2.0`; report mean, max, and configured percentiles (`95` by default).                                                | Matched prediction/GT boxes                            | Per-class `corner_mean`, `corner_p95`, `corner_max`; macro `mcorner_mean`, `mcorner_max` |
| **Heading-flip rate (A2)**              | A TP is flipped when its absolute wrapped yaw error is greater than `pi/2`.                                                                                                                                                                                                                                  | Matched prediction/GT yaw                              | Per-class flip count/rate; macro `mflip_rate`                                            |
| **Signed nearest-surface error (A3)**   | `nearest_surface_distance(pred) - nearest_surface_distance(gt)`, where the distance is from ego to the closest BEV footprint point. Positive means the predicted near face is too far away. Defaults: p5/p95 tails.                                                                                          | Matched prediction/GT footprints                       | Per-class mean, low, high, absolute max; macro high and absolute max                     |
| **Critical FP/FN (B1)**                 | At confidence operating points `0.3` and `0.5`, class-agnostically match boxes at 2.0 m. Count unmatched predictions and GT boxes with finite TTC. Divide each count by the number of TTC-covered frames.                                                                                                    | Per-frame raw boxes, scores, labels, TTC, map coverage | Overall and per-class critical FP/frame and FN/frame at each confidence                  |
| **Collision-weighted mAP (B2)**         | Weight every object by `w = exp(-decay * TTC)`, default `decay=0.5`. A TP inherits its matched GT weight; an FP keeps its predicted-object weight. Unreachable objects have weight zero. Compute nuScenes-style interpolated AP at thresholds `(0.5, 1.0, 2.0, 4.0)`, then mean over thresholds and classes. | Per-frame raw boxes, scores, labels, TTC, map coverage | `cw_mAP` and per-class `cw_mAP_<class>`                                                  |
| **Detection calibration error (E1)**    | Expected calibration error over equal-width score bins (`num_bins=15`). Correctness is TP status at 2.0 m. Validate that scores are probabilities in `[0, 1]`.                                                                                                                                               | Prediction scores and TP flags                         | Pooled `ece` and mean per-predicted-class `ece_macro`                                    |
| **Detection confident-error rate (E3)** | Among FPs at or above reporting floor `min_score=0.1`, calculate the fraction at or above `score_threshold=0.5`. FNs have no score and are intentionally excluded.                                                                                                                                           | Prediction scores and FP flags at 2.0 m                | Rate, count, and count/frame                                                             |
| **Detection confusion matrix**          | Class-agnostically match predictions above `min_score=0.1` at 2.0 m. Count matched `(true_class, predicted_class)` pairs only; unmatched boxes are not matrix cells.                                                                                                                                         | Per-frame raw boxes, scores, labels                    | Raw flattened cells `confusion_<true>__<pred>`                                           |

For B2, reuse the current `Ap` convention exactly: 101 recall points, ignore recall through 0.1,
subtract minimum precision 0.1, clamp at zero, and normalize by 0.9. Add golden tests proving that
unit weights produce the same result as unweighted AP.

### 4.2 Point-cloud segmentation metrics

The segmentation state is frame-oriented. Each frame supplies coordinates, target class, predicted
class, and class probabilities. Invalid/ignored targets are excluded before every metric.

| Metric                                     | Definition and defaults                                                                                                                                                                                                       | Required outputs                                                            |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Segmentation confusion matrix**          | Point-level raw counts, rows=true and columns=prediction.                                                                                                                                                                     | `confusion_<true>__<pred>`                                                  |
| **Segmentation calibration error (E1)**    | ECE with 15 confidence bins; confidence is the probability assigned to the reported predicted class. The classwise macro rows are selected by predicted class.                                                                | `ece`, `ece_macro`                                                          |
| **Uncertainty usefulness (E2)**            | AUROC of normalized predictive entropy as a detector of misclassified points. Use fixed histograms (`8192` bins) and tie-aware AUROC to keep memory bounded.                                                                  | `entropy_auroc`, mean entropy for wrong/correct points                      |
| **Segmentation confident-error rate (E3)** | Fraction of misclassified points whose normalized entropy is below `0.3`.                                                                                                                                                     | Rate and count                                                              |
| **Error clusters (D1)**                    | Strict point error rate plus connected components of the radius-neighbor graph over error points. Defaults: `cluster_radius=0.5 m`, `min_cluster_points=1`.                                                                   | Overall/per-class error rates and cluster counts; clusters/frame            |
| **Partial-detection score (D2)**           | For mapped small-object GT boxes, take valid segmentation points inside the yaw-aware BEV footprint. With `k` correct among `n`, credit is `(k/(k+h)) / (n/(n+h))`, default `h=1`. Skip boxes with fewer than `min_points=1`. | Mean partial-detection score overall/per detection class; skipped-box count |
| **Neighborhood-tolerant error rate (D3)**  | A wrong point is rescued if any point within `radius=0.2 m` predicts its true class. At radius zero this must equal strict error rate.                                                                                        | Overall/per-class tolerant error rates and counts                           |

Entropy must be computed as `-sum(p * log(p)) / log(C)` with zero-probability terms treated as
zero. Inputs named `scores` in the source are probabilities, not logits; the public contract in
`perception_eval` should call them `probabilities` and reject negative values, values above one,
rows that do not sum to one within tolerance, and a class-column count that differs from the class
configuration. Confidence is gathered from the reported predicted-class column. By default,
validation should also require the reported prediction to equal `argmax(probabilities)`; if a
caller deliberately disables that check, the gathered probability remains the confidence.

## 5. Current `perception_eval` Gaps

The following gaps affect the design:

- `MetricsScore` is a fixed aggregator for detection, tracking, prediction, and classification;
  there is no component registry or generic report namespace.
- `PerceptionFrameResult` retains matched results and filtered GT, but does not retain the complete
  filtered prediction list needed for class-agnostic rematching and filter-specific evaluation.
- scene aggregation flattens matches across frames. B1, B2, confusion, TTC coverage, and
  clusters/frame require frame boundaries to remain available.
- `visualization/detection_confusion_matrix.py` already draws a matrix with unmatched FP/FN cells,
  but it consumes label-partitioned existing matches. The new source metric is a different,
  class-agnostically matched diagnostic containing only matched class pairs; the shared report
  should become the visualization's data source only after this semantic difference is made
  explicit.
- `FrameGroundTruth` has transforms but no stable `scene_id`/map reference.
- the repository supports point-cloud loading for sensing, but not semantic point targets,
  probabilities, or a segmentation evaluation task.
- Shapely, SciPy, and NumPy are already transitive/direct project dependencies, so the source
  geometry and KD-tree algorithms can be implemented without introducing PyTorch. SciPy should be
  declared directly if the new public metrics import it.

## 6. Target Architecture

### 6.1 Package layout

Add the following modules. Names are intentionally aligned with the source project where practical
so formula parity is easy to audit.

```text
perception_eval/perception_eval/evaluation/metrics/
├── component.py                 # MetricComponent, MetricFilter, MetricRange, MetricReport
├── class_groups.py              # taxonomy validation and label/confusion folding
├── confusion_report.py          # stable class tokens and flattened cells
├── filters.py                   # Identity, Region, Corridor, Collision filters
├── geometry/
│   ├── lanelet.py               # OSM parser, LaneletMap, provider/resolver
│   └── reachability.py          # Agent, parameters, reachable sets, TTC, weights
├── detection/
│   ├── state.py                 # DetectionFrame, lazy curves, filter/range views
│   ├── matching.py              # score-ordered greedy matching helpers
│   ├── geometry.py              # corner and nearest-surface helpers
│   ├── calibration.py
│   ├── confident_error.py
│   ├── confusion_matrix.py
│   ├── corner_error.py
│   ├── heading_flip.py
│   ├── nearest_surface_error.py
│   ├── collision.py
│   ├── critical_fp_fn.py
│   └── collision_weighted_map.py
└── segmentation/
    ├── state.py                 # SegmentationFrame and derived confidence/entropy
    ├── suite.py                 # confusion-backed and raw-point evaluation paths
    ├── spatial.py               # tolerant mask and error-cluster helpers
    ├── calibration.py
    ├── confident_error.py
    ├── confusion_matrix.py
    ├── entropy_auroc.py
    ├── error_clusters.py
    ├── partial_detection.py
    └── tolerant_error.py
```

Use existing `detection/ap.py`, `detection/map.py`, and `detection/tp_metrics.py` rather than
forking the baseline AP implementation.

### 6.2 Core interfaces

Implement small, non-framework-specific protocols:

```python
@dataclass(frozen=True)
class MetricRange:
    name: str
    min_distance: float = 0.0
    max_distance: float | None = None

class MetricFilter(Protocol):
    name: str
    required_context: tuple[str, ...]
    def available(self, context: FrameMetricContext) -> bool: ...
    def keep_points(self, xyz: NDArray, context: FrameMetricContext) -> NDArray[np.bool_]: ...
    def keep_objects(self, objects: Sequence[DynamicObject], context: FrameMetricContext) -> NDArray[np.bool_]: ...

class MetricComponent(Protocol[StateT]):
    def evaluate(self, state: StateT) -> dict[str, float]: ...

@dataclass
class MetricReport:
    values: dict[str, float]
    coverage: dict[str, tuple[int, int]]
```

`MetricSuite` should build a view for each distinct `(taxonomy, filter, range)` combination, call
only the components configured for that view, and prefix keys deterministically. Do not mutate
`DynamicObject` instances when filtering or folding labels.

Recommended key order:

```text
<task>/<taxonomy?>/<filter?>/<range?>/<metric-key>
```

Examples: `detection/corner_mean_car`,
`detection/region_road/0_30/corner_p95_car`, and
`segmentation/grouped/region_road/error_clusters_per_frame`. Preserve an optional legacy flat-key
formatter if downstream log consumers cannot accept slashes.

### 6.3 Detection frame contract

Add an immutable scene-metric input retained by `PerceptionFrameResult`:

```python
@dataclass(frozen=True)
class DetectionFrame:
    frame_name: str
    scene_id: str | None
    estimated_objects: tuple[DynamicObject, ...]
    ground_truth_objects: tuple[DynamicObject, ...]
    transforms: TransformDict
```

`PerceptionEvaluationManager.add_frame_result()` should populate it after the existing common
filtering step and before matching. `get_scene_result()` should pass the ordered frame list to the
new detection suite in addition to running the existing mAP path.

Use `TransformDict[(FrameID.BASE_LINK, FrameID.MAP)]` as the canonical pose source. Geometry
adapters must normalize every object to `base_link` exactly once and raise a clear error when a
required transform is absent.

The box adapter must explicitly translate the local convention
`Shape.size == (width, length, height)` into the source convention
`[dx=length, dy=width, dz=height]`; silently copying the tuple would rotate dimensions and corrupt
A1, A3, region overlap, and TTC body radii. Initially require `ShapeType.BOUNDING_BOX` for metrics
whose formula assumes four box corners. If polygon-shaped detections must be supported later,
define and test a separate polygon formula rather than approximating them implicitly.

Extend `FrameGroundTruth` with optional `scene_id` and preserve it in reduce/serialization. Populate
it from the nuScenes/T4 sample record. Deserialization must accept old files where the field is
missing.

### 6.4 Segmentation frame contract

Introduce a model-independent public input:

```python
@dataclass(frozen=True)
class SegmentationFrame:
    frame_name: str
    scene_id: str | None
    coordinates: NDArray[np.float32]       # (N, 3+) in base_link
    targets: NDArray[np.int64]             # (N,)
    predictions: NDArray[np.int64]         # (N,)
    probabilities: NDArray[np.float32]     # (N, C)
    transforms: TransformDict
    gt_objects: tuple[DynamicObject, ...] = ()
```

Provide a dedicated `SegmentationEvaluationManager` rather than overloading
`PerceptionEvaluationManager.add_frame_result()` with mutually incompatible inputs. The first
delivery may accept already-decoded NumPy arrays; dataset/model adapters can be added separately.

Maintain two accumulation strategies:

- confusion-backed metrics accumulate fixed-size integer matrices and do not retain all points;
- calibration, entropy, cluster, tolerant-error, and partial-detection metrics process one frame
  at a time into bounded sufficient statistics whenever possible. D1/D3 need only the current
  frame, so the offline evaluator does not need the source project's epoch-long raw-point cache.

### 6.5 Map and coverage contract

Port the Lanelet2 parser with the following behaviors:

- parse local `local_x`/`local_y` nodes;
- form lanelet polygons from left and reversed-right bounds;
- recognize the source set of lanelet subtypes and area-way types;
- parse `speed_limit` as km/h and convert to m/s;
- repair invalid polygons with `buffer(0)` and ignore empty/zero-area polygons;
- cache parsed maps by resolved absolute OSM path;
- choose the lowest speed in overlapping lanelets;
- support point containment and box-footprint intersection;
- keep `available(scene_id)` separate from `get(scene_id)`: a missing map excludes only the
  map-dependent view, while a corrupt map that claims to exist raises.

The source uses Shapely 2 vectorized APIs, while this repository intentionally supports Shapely 1
on Python 3.10. Put version differences behind a small compatibility module: Shapely 2 may use
vectorized `points`/`contains`/`intersects`, while Shapely 1 uses prepared geometries and explicit
loops. Handle the different `STRtree.query()` return types in the same module and run map tests in
both dependency environments. Do not copy Shapely 2-only calls into metric code.

Support an explicit configuration mapping first:

```yaml
metric_maps:
  resolver: t4_scene_directory
  data_root: /path/to/t4/data
```

The resolver maps `<data_root>/<scene_id>/map/lanelet2_map.osm`. An alternative explicit
`scene_id -> osm_path` mapping should be available for non-T4 datasets and tests.

Every map-dependent report must include coverage `(covered_frames, seen_frames)`. If coverage is
zero, report `NaN`, never zero. If coverage is partial, exclude uncovered frames from only that
view and emit one warning per scene report.

## 7. Detailed Implementation Sequence

### Phase 0: Freeze upstream behavior

1. Copy/adapt the pure-math test vectors from the source comparison into a new
   `perception_eval/test/evaluation/metrics/` fixture area.
2. Record the source commit `fcf86419` and retain Apache-2.0 attribution in ported files.
3. Create golden JSON fixtures for metric keys, default parameters, empty inputs, and NaN rules.
4. Add an explicit parity matrix linking every source test to its local equivalent.

Exit criterion: every metric and helper in Section 4 has at least one independent expected-value
fixture before implementation begins.

### Phase 1: Shared component, naming, taxonomy, and range primitives

1. Implement `MetricRange`, component/report interfaces, deterministic key composition, duplicate
   key detection, and finite/NaN formatting.
2. Implement full-partition class groups. Validate missing class names, duplicate membership,
   unassigned trained classes, empty groups, and output-token collisions.
3. Implement label and confusion-matrix folding. Out-of-range ignore labels remain unchanged.
4. Add range views using BEV center distance for boxes and BEV radial distance for points.

Exit criterion: grouped and ungrouped toy confusion matrices produce the source results; filter and
range key collisions fail at configuration time.

### Phase 2: Detection scene state and matching

1. Retain `DetectionFrame` in every frame result and through serialization.
2. Port stable score-ordered greedy matching. Compute the distance matrix and candidate ordering
   once per `(frame, taxonomy)` and reuse them across thresholds.
3. Build lazy per-class match curves containing score, TP/FP, matched GT index, yaw error, corner
   error, and signed near-surface error.
4. Compare local curves against current `NuscenesObjectMatcher`/`Map` on fixture scenes. Document
   any tie-breaking difference; use stable prediction-score order and stable nearest-GT index as
   the normative rule.
5. Keep current `evaluate_detection()` untouched until the parity comparison passes. New component
   reports live in `MetricsScore.detection_metric_report`.

Exit criterion: the new state reproduces existing center-distance AP/mAPH within `1e-12` on unit
fixtures and within an agreed numerical tolerance on a real scene.

### Phase 3: Detection diagnostics without maps

Implement in this order:

1. geometry helpers and A1 corner displacement;
2. A2 heading-flip rate;
3. A3 signed nearest-surface error;
4. E1 detection calibration;
5. E3 detection confident-error rate;
6. class-agnostic matched-pair confusion matrix;
7. corridor filtering, because it is map-independent.

All components must handle empty predictions, empty GT, and classes with no TP according to the
source NaN/count semantics. Add range-sliced tests for boxes whose center is outside a filter but
whose footprint overlaps it.

Exit criterion: all six non-map detection metrics pass source-derived golden tests, and enabling
them does not change existing mAP output.

### Phase 4: Lanelet region filters

1. Port the OSM parser, spatial indexes, cached unions, erosion/expansion behavior, and speed lookup.
2. Implement `RegionFilter` and `CorridorFilter` separately from the TTC model.
3. Add frame coverage accounting and warning/report serialization.
4. Test known-but-absent map regions as empty slices, unknown region tokens as configuration
   errors, map-less scenes as uncovered, and corrupt available maps as hard failures.

Exit criterion: region point membership and object-footprint overlap match the source map fixture,
including inward and outward margins.

### Phase 5: Reachability, collision filter, and criticality metrics

1. Port `ReachabilityParams` with defaults: horizon `4.0 s`, step `0.1 s`, max lateral acceleration
   `3.0 m/s^2`, minimum turn radius `3.0 m`, and `21` arc samples.
2. Port the `STATIC`, `VRU`, and `WHEELED` reachable-set models and ego-frame reuse.
3. Implement the detection-class adapter. Require an explicit mapping for every configured class;
   provide the source defaults for car/truck/bus/train/motorcycle, pedestrian/animal/bicycle, and
   static obstacles. Defaults for VRU speeds are 3/4/6 m/s and off-map wheeled fallback is 16.7 m/s.
4. Compute predicted and GT TTC once per covered frame and cache by frame plus reachability config.
5. Implement `CollisionFilter`, B1 Critical FP/FN, and B2 collision-weighted mAP.
6. Benchmark the pre-screen and cached ego reachability on scenes with many objects. TTC must not be
   recalculated for each confidence, threshold, filter, or metric.

Exit criterion: source scenarios pass (same-speed lead is unreachable, stationary lead and
oncoming/crossing agents have finite TTC, disconnected road is unreachable), collision weights are
monotonic, and map coverage controls denominators correctly.

### Phase 6: Segmentation foundation

1. Add `EvaluationTask.SEGMENTATION` only after auditing all `EvaluationTask` conditionals and
   frame-ID defaults; use a separate manager to keep detection behavior isolated.
2. Add `SegmentationMetricsConfig`, validation, serialization, and the `SegmentationFrame` contract.
3. Implement probability validation, confidence, normalized entropy, valid-point masks, per-filter
   masks, range buckets, and class-group views.
4. Implement fixed-size confusion accumulation first, then expose ordinary accuracy/IoU/F1 from the
   same matrix if desired. Those baseline segmentation metrics are useful acceptance checks but are
   not new metrics from this source range.
5. Define memory budgets and expose frame-by-frame `update()`/`compute()` so large point clouds are
   never concatenated across a scene.

Exit criterion: a multi-frame toy scene produces identical whole-scene, range, region, and grouped
confusion matrices regardless of frame update order.

### Phase 7: Point-cloud segmentation metrics

1. Implement E1 calibration and E3 confident-error from per-frame sufficient statistics.
2. Implement E2 with wrong/correct entropy histograms; validate against an exact rank-based AUROC on
   small fixtures and bound quantization error by one bin width.
3. Implement SciPy `cKDTree` helpers for D3 neighbor rescue and D1 radius-graph connected components.
4. Implement D1 strict error/cluster aggregation per class and per frame.
5. Implement D3 tolerant-error aggregation and prove `radius=0` equals strict error rate.
6. Implement D2 yaw-aware box membership, saturating credit, class-space validation, low-point skip
   accounting, and filter application to both points and GT boxes.
7. Apply taxonomy folding before uncertainty derivation for grouped metrics: grouped confidence is
   the summed probability mass of the predicted group, not the maximum member-class probability.

Exit criterion: all source-derived point-metric tests pass without retaining more than one frame's
coordinates in memory.

### Phase 8: Public integration and documentation

1. Add opt-in component lists to `DetectionMetricsConfig` and `SegmentationMetricsConfig`.
2. Add typed report accessors and concise `__str__` summaries; keep raw confusion cells available in
   serialization rather than printing the full matrix by default.
3. Document all formulas, units, signs, thresholds, key prefixes, coverage, NaN behavior, and example
   configurations in `docs/en/perception/metrics.md` and the Japanese counterpart.
4. Add an end-to-end example using one map-covered and one map-less scene.
5. Run formatting, unit tests, serialization round trips, performance benchmarks, and the existing
   full test suite.

Exit criterion: a user can enable any metric through configuration, consume a stable serialized
report, and understand why a value is NaN or based on partial coverage.

## 8. Configuration Proposal

Preserve all existing flattened detection threshold keys. Add one optional nested field parsed by
`DetectionMetricsConfig`; absence means legacy behavior only. Class names use the `AutowareLabel`
vocabulary of this repository (`car, truck, bus, motorbike, bicycle, pedestrian, animal, hazard,
unknown`); traffic cones and barriers are `hazard` (or `unknown` when `merge_similar_labels` is
enabled).

```yaml
advanced_detection_metrics:
  ranges:
    - { name: 0_30, min_distance: 0.0, max_distance: 30.0 }
    - { name: 30_60, min_distance: 30.0, max_distance: 60.0 }
  class_groups:
    grouped_vehicle: [car, truck, bus]
    grouped_vru: [pedestrian, bicycle, motorbike]
    grouped_static: [hazard, unknown]
  filters:
    - { name: corridor, type: corridor, width_m: 3.0 }
    - { name: road, type: region, regions: [road, road_shoulder, crosswalk] }
    - { name: collision, type: collision }
  components:
    - { type: corner_error, tp_threshold: 2.0, percentiles: [95.0] }
    - { type: heading_flip, tp_threshold: 2.0, flip_threshold: 1.57079632679 }
    - { type: nearest_surface_error, tp_threshold: 2.0 }
    - { type: calibration, tp_threshold: 2.0, num_bins: 15 }
    - { type: confident_error, tp_threshold: 2.0, min_score: 0.1, score_threshold: 0.5 }
    - { type: confusion_matrix, match_threshold: 2.0, min_score: 0.1 }
    - { type: critical_fp_fn, confidences: [0.3, 0.5], match_threshold: 2.0 }
    - { type: collision_weighted_map, thresholds: [0.5, 1.0, 2.0, 4.0], decay: 0.5 }
  map:
    resolver: t4_scene_directory
    data_root: /path/to/t4/data
```

Do not instantiate arbitrary Python classes from configuration. Use a closed registry from `type`
tokens to validated dataclass configurations and component constructors.

## 9. Test Plan

### Unit tests

- class-group partition validation and folded labels/confusion;
- stable key tokens and duplicate detection;
- corner assignment, yaw wrapping, nearest-surface sign;
- greedy matching across empty inputs, score ties, distance ties, and multiple thresholds;
- ECE hand calculations and invalid probability inputs;
- confident-error floors/thresholds and empty denominators;
- weighted AP landmarks and unit-weight parity;
- Lanelet OSM parsing, regions, margins, footprint overlap, and speed limits;
- reachable sets, TTC pre-screen, collision weights, and class adapter validation;
- entropy normalization and histogram AUROC versus exact AUROC;
- KD-tree tolerant errors and clusters versus brute-force references;
- partial-detection credit at `k=0`, `k=1`, and `k=n`;
- coverage zero/full/partial behavior and warnings.

### Integration tests

- frame result and scene result serialization, including old data without `scene_id`;
- existing detection scene evaluation with advanced metrics disabled;
- detection scene with unfiltered, range, grouped, region, corridor, and collision views;
- segmentation multi-frame streaming evaluation with point counts large enough to reveal accidental
  scene-wide concatenation;
- missing transforms, missing maps, corrupt OSM, unknown classes, and class-space mismatch failures;
- output key snapshot tests to make accidental API changes explicit.

### Performance tests

Measure peak RSS and wall time separately for:

- baseline detection;
- non-map advanced detection metrics;
- map filters;
- TTC/B1/B2;
- confusion-only segmentation;
- each point-level segmentation metric.

Suggested acceptance targets are less than 10% overhead when advanced metrics are disabled, no
scene-size growth for segmentation memory, one OSM parse per map, and one ego-reachability build per
covered frame/configuration.

## 10. Recommended Pull Request Breakdown

| PR  | Deliverable                                                                   | Depends on    |
| --- | ----------------------------------------------------------------------------- | ------------- |
| 1   | Shared component/report API, naming, taxonomy, range helpers, golden fixtures | None          |
| 2   | Detection frame retention, matching state, geometry helpers                   | PR 1          |
| 3   | A1/A2/A3, detection E1/E3, detection confusion, corridor filter               | PR 2          |
| 4   | Lanelet parser/provider, region filter, coverage reporting                    | PR 1          |
| 5   | Reachability/TTC, collision filter, B1/B2                                     | PRs 2 and 4   |
| 6   | Segmentation task/config/manager, frame contract, confusion suite             | PRs 1 and 4   |
| 7   | Segmentation E1/E2/E3 and D1/D2/D3                                            | PR 6          |
| 8   | End-to-end examples, bilingual docs, benchmarks, default-policy decision      | All prior PRs |

PRs 2 and 4 can be developed in parallel after PR 1. Keep segmentation support separate from the
detection PRs because it introduces a new public task and data contract.

## 11. Effort Estimate

### 11.1 Assumptions

One person-day (PD) is eight engineering hours. The estimates include implementation, local design,
unit and integration tests, backward-compatibility work, code-review fixes, and the documentation
listed in this plan. They assume an engineer who is already familiar with Python and
`perception_eval`, access to at least one representative T4 scene with a Lanelet2 map, and timely
answers to the decisions in Section 12.

They exclude organizational waiting time, dataset access/setup outside this repository, CI machine
provisioning, dashboard or Web.Auto integration, production threshold tuning, and changes requested
because the upstream metric specification changes after `fcf86419`.

### 11.2 Estimate by pull request

| PR  | Work item                                                                | Estimate (PD) | Main uncertainty                                        |
| --- | ------------------------------------------------------------------------ | ------------: | ------------------------------------------------------- |
| 1   | Shared component/report API, naming, taxonomy, ranges, golden fixtures   |           4-6 | Final public key/config format                          |
| 2   | Detection frame retention, matching state, geometry, serialization       |           5-8 | Parity with current matching and old serialized results |
| 3   | Non-map detection metrics and corridor filtering                         |           6-9 | Edge cases and output parity across ranges/groups       |
| 4   | Lanelet parser/provider, Shapely compatibility, region filters, coverage |          6-10 | T4 map variants and Shapely 1/2 behavior                |
| 5   | Reachability/TTC, collision filter, B1/B2, performance work              |         10-16 | Geometry correctness and runtime on real scenes         |
| 6   | Segmentation task/config/manager, frame contract, confusion suite        |          8-13 | New public task and source of segmentation input arrays |
| 7   | Segmentation E1/E2/E3 and D1/D2/D3                                       |         10-16 | Large-cloud performance and cross-task D2 inputs        |
| 8   | End-to-end examples, bilingual docs, benchmarks, stabilization           |           4-7 | Findings from real-data validation                      |
|     | **Implementation subtotal**                                              |     **53-85** |                                                         |

Reserve 20% for integration risk, real-data discrepancies, and review feedback. The recommended
project budget is therefore **64-102 PD**, with **approximately 80 PD** as the planning point.

### 11.3 Scope-level estimates

| Deliverable                                                            | Included PRs | Implementation | Budget with 20% reserve |
| ---------------------------------------------------------------------- | ------------ | -------------: | ----------------------: |
| Detection MVP without maps/TTC                                         | 1-3          |       15-23 PD |                18-28 PD |
| Complete detection metrics                                             | 1-5          |       31-49 PD |                38-59 PD |
| Segmentation foundation and metrics, incremental after shared/map work | 6-7          |       18-29 PD |                22-35 PD |
| Full scope including stabilization                                     | 1-8          |       53-85 PD |               64-102 PD |

The recommended first release is the **detection MVP without maps/TTC**. It validates the component
API, frame retention, matching parity, and metric reporting before the two highest-risk areas
(reachability geometry and a new segmentation task) are added.

### 11.4 Indicative calendar duration

Calendar duration is longer than `total PD / engineer count` because PR 1 is a common dependency,
PR 5 depends on both PRs 2 and 4, and PR 8 follows all feature work.

| Staffing    | Expected elapsed time | Notes                                                                                        |
| ----------- | --------------------- | -------------------------------------------------------------------------------------------- |
| 1 engineer  | 16-26 weeks           | Lowest coordination cost; all work is sequential                                             |
| 2 engineers | 10-16 weeks           | Parallelize PR 2 with PR 4, then PR 5 with PRs 6-7 where dependencies allow                  |
| 3 engineers | 8-14 weeks            | Best practical throughput; more staffing is unlikely to shorten the critical path materially |

These durations assume roughly four productive implementation days per engineer per week, with the
remaining time covering review, meetings, CI turnaround, and iteration on real-data findings.

### 11.5 Estimate adjustments

Apply these adjustments once the Section 12 decisions are made:

- subtract 2-3 PD if Shapely 2 becomes the only supported version;
- subtract 4-7 PD from PR 6 if segmentation input is supplied exclusively by an external adapter
  and no dataset-loading integration is required;
- add 3-5 PD if the advanced stable greedy matcher cannot reproduce current center-distance AP and
  both matching behaviors must remain supported and explained;
- add 3-6 PD if polygon-shaped detection objects must be supported in A1/A3/TTC in the first release;
- add 5-10 PD if metric outputs must also be integrated into an external dashboard or release-gate
  system;
- omit PRs 6-7 and reduce PR 8 by approximately 1-2 PD if segmentation is explicitly out of scope.

## 12. Decisions Required Before Coding

Resolve these items in PR 1; they change public behavior or dataset integration:

1. **Metric key format**: slash-separated canonical keys versus legacy-compatible flat keys.
2. **Scene/map identity**: confirm the T4 `scene_id` value stored in annotation data is the relative
   directory fragment expected by the OSM resolver.
3. **Detection matching parity**: decide whether the new stable greedy matcher becomes the eventual
   implementation behind existing mAP or remains an advanced-metric-only engine.
4. **Class taxonomy ownership**: define project-level default groups or require every caller to
   provide a full partition.
5. **Segmentation input ownership**: determine whether this repository will load labeled
   segmentation point clouds or only evaluate arrays supplied by an external adapter.
6. **SciPy dependency**: promote SciPy to a direct dependency or implement slower NumPy fallbacks.
7. **Shapely support**: retain Shapely 1 compatibility on Python 3.10 through wrappers, or make a
   separately reviewed dependency-policy change to require Shapely 2 everywhere.
8. **Release gating**: B1 is suitable for separate safety/usability gates; D2 is diagnostic only;
   B2 must always be shown alongside unweighted mAP. Confirm which, if any, are enabled in standard
   reports after validation.

## 13. Definition of Done

The port is complete when:

- every metric in Section 4 is configurable and documented;
- formulas, defaults, empty-data behavior, and output keys match the source commit or have a
  documented intentional difference;
- existing mAP/mAPH values and existing serialized inputs remain backward compatible;
- all taxonomy/filter/range combinations are deterministic and collision-free;
- map-dependent metrics expose full/partial/zero coverage and never convert missing coverage into a
  misleading zero;
- segmentation evaluates in streaming fashion with bounded memory;
- upstream-derived golden tests, new integration tests, and the existing repository test suite pass;
- performance measurements and an example result are attached to the final integration PR.
