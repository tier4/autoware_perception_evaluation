# Advanced detection metrics: file contracts (schema v1)

The driving-aware detection metrics ([metrics.md](metrics.md))
are computed per scenario and pooled across scenarios afterwards. Two files per scenario make that
possible: a per-scenario **report** (`advanced_detection_metrics.json`) and the **prepared samples**
(`advanced_detection_samples.npz`). `perception_eval` owns the sample file format
(`PreparedSamples.to_npz` / `from_npz`) and the report's `config_digest`, `values`, `coverage`,
`counts` and `warnings`; the report file itself is written by downstream log-replay tools (for
example `driving_log_replayer_v2`), which also add the producer and scene fields. Pooling tools read
both files back and call `evaluate_prepared(config, parts)`.

Both files are written atomically (write `<name>.tmp`, then `os.replace`) into
`result_archive/<topic.with.dots>/` next to `scene_result.pkl`. They are written only when
`advanced_detection_metrics` is configured and the evaluation task is not `fp_validation`, from the
frames **after** ignored-frame removal, from **one** suite run (`prepare` once, `evaluate_prepared`
once).

## Versioning and digests

- `PREPARED_SCHEMA_VERSION = 1` (`perception_eval.evaluation.metrics.detection.prepared`). Readers
  reject any other `schema_version` with `PoolingError`.
- `config_digest(config)` = `sha256:<hex>` of the canonical JSON (`sort_keys=True`,
  `separators=(",", ":")`) of
  `{"class_names": [...], "advanced": <config.serialization() without map.data_root and map.mapping>, "prepared_schema_version": 1}`.
  It is invariant to where the maps live and to dict order; it changes with any component, filter,
  range, class group, matching or score parameter.
- `frame_filter_digest` (report only) = sha256 of the canonical JSON of the
  `evaluation_config_dict` keys `target_labels`, `max_x_position`, `max_y_position`, `max_distance`,
  `min_distance`, `min_point_numbers`, `confidence_threshold`, `label_prefix`,
  `merge_similar_labels`, `allow_matching_unknown` (a missing key is `null`). Scenarios whose frame
  filter differs were evaluated on different object sets and are not pooled together.

## `advanced_detection_samples.npz`

Written with `np.savez_compressed`; contains no object arrays and is read with
`np.load(path, allow_pickle=False)`. `F` frames, `P` predictions, `G` ground truths,
`K = len(filter_names)`. `F = 0` is valid (an empty scene).

| key                          | dtype / shape                    | notes                                                                                                                                                                                                                                                   |
| ---------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `header`                     | `uint8[N]`                       | UTF-8 JSON: `{"schema_version": 1, "config_digest": "sha256:...", "class_names": [...], "filter_names": [...], "has_ttc": bool, "box_columns": ["cx","cy","cz","dx","dy","dz","yaw","vx","vy"], "counts": {...}, "warnings": [...], "producer": {...}}` |
| `frame_name`                 | `<U64 [F]`                       | frame names in input order (skipped frames are absent)                                                                                                                                                                                                  |
| `unix_time`                  | `int64[F]`                       |                                                                                                                                                                                                                                                         |
| `ttc_covered`                | `bool[F]`                        | False when the scene had no lanelet map or the frame no ego pose                                                                                                                                                                                        |
| `available`                  | `bool[F, K]`                     | per frame and non-identity filter (column order = `filter_names`)                                                                                                                                                                                       |
| `pred_offsets`, `gt_offsets` | `int64[F + 1]`                   | CSR row offsets: frame `i` owns rows `offsets[i]:offsets[i + 1]`                                                                                                                                                                                        |
| `pred_boxes`, `gt_boxes`     | `float64[P, 9]`, `float64[G, 9]` | base_link rows `[cx, cy, cz, dx(length), dy(width), dz(height), yaw, vx, vy]`                                                                                                                                                                           |
| `pred_scores`                | `float64[P]`                     | the configured `score_source`; non-finite scores are already dropped                                                                                                                                                                                    |
| `pred_labels`, `gt_labels`   | `int64[P]`, `int64[G]`           | index into `class_names`                                                                                                                                                                                                                                |
| `pred_ttc`, `gt_ttc`         | `float64[P]`, `float64[G]`       | only when `has_ttc`; reachability time-to-collision in seconds, `inf` = unreachable                                                                                                                                                                     |
| `pred_masks`, `gt_masks`     | `bool[P, K]`, `bool[G, K]`       | elements each filter keeps; all-False rows for frames where the filter is unavailable                                                                                                                                                                   |

`counts` holds `input_frames`, `skipped_frames`, `polygon_as_box`, `non_finite_score`, `pred_total`,
`pred_kept` and `zero_score`. `producer` is free-form (`{"name": ..., "version": ...}`), `{}` when
not given. `to_npz` returns the sha256 hex digest of the written file; the report records it.

## `advanced_detection_metrics.json` (per scenario)

Strict JSON (`json.dump(..., allow_nan=False)`): every `NaN` value is written as `null`.

```json
{"schema":"perception_eval.advanced_detection_report","schema_version":1,
 "status":"ok|empty|error","error":null,
 "producer":{"name":"driving_log_replayer_v2","version":"3.x.y"},
 "perception_eval":{"version":"1.4.0","prepared_schema_version":1},
 "topic":"/perception/object_recognition/objects","evaluation_task":"tracking","frame_id":"map",
 "scene":{"dataset_name":"<basename of t4_dataset_path>","map_available":true},
 "config_digest":"sha256:...","frame_filter_digest":"sha256:...",
 "config":{...AdvancedDetectionMetricsConfig.serialization() minus map paths, plus class_names...},
 "frames":{"frame_results":612,"ignored":20,"evaluated":592,"skipped":0,"prepared":592},
 "counts":{"pred_total":0,"pred_kept":0,"non_finite_score":0,"polygon_as_box":0,"zero_score":0,"skipped_frames":0},
 "coverage":{"road":[592,592],"collision":[592,592],"ttc":[592,592]},
 "values":{"detection/corner_mean_car":0.41},
 "warnings":[],
 "samples_file":{"name":"advanced_detection_samples.npz","sha256":"...","num_frames":592}}
```

| field                                  | meaning                                                                                                                                                            |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `schema`, `schema_version`             | `perception_eval.advanced_detection_report`, `1`                                                                                                                   |
| `status`, `error`                      | `ok`; `empty` when no frame survived preparation; `error` with the message in `error` when the suite raised (then `values` is `{}` and no samples file is written) |
| `producer`                             | the tool that wrote the file and its version                                                                                                                       |
| `perception_eval`                      | `version` of this package and the `prepared_schema_version` of the samples file                                                                                    |
| `topic`, `evaluation_task`, `frame_id` | the evaluated topic, task (`detection`/`tracking`) and the objects' frame id                                                                                       |
| `scene`                                | `dataset_name` (basename of the T4 dataset path) and whether a lanelet map was resolved                                                                            |
| `config_digest`, `frame_filter_digest` | see above; pooling tools group scenarios by both                                                                                                                   |
| `config`                               | `AdvancedDetectionMetricsConfig.serialization()` without `map.data_root` / `map.mapping`, plus `class_names`                                                       |
| `frames`                               | `frame_results` received, `ignored` (removed before evaluation), `evaluated` (= handed to `prepare`), `skipped` by the suite, `prepared` (= `evaluated - skipped`) |
| `counts`                               | `PreparedSamples.counts` (same keys)                                                                                                                               |
| `coverage`                             | `MetricReport.coverage` as `[covered, seen]` pairs                                                                                                                 |
| `values`                               | `MetricReport.values` for this scenario alone (`null` = NaN); pooled values are recomputed from the samples, never averaged from these                             |
| `warnings`                             | `MetricReport.warnings`                                                                                                                                            |
| `samples_file`                         | name, sha256 and `num_frames` of the samples file next to the report                                                                                               |

## Pooling

```python
from perception_eval.evaluation.metrics.detection.prepared import PreparedSamples, evaluate_prepared

parts = [PreparedSamples.from_npz(path) for path in paths]  # canonical order, see below
report = evaluate_prepared(config, parts)  # PoolingError on digest / class / filter mismatch
```

- Canonical pooling order: scenarios sorted by `(dataset_name, scenario_path)`, frames in file
  order. With that order `evaluate_prepared(parts)` is bit-identical to one `suite.evaluate` over the
  concatenated frames. Equal scores are matched in input order (stable sort), so a different part
  order may change tie-breaks; it never changes which frames are pooled.
- `evaluate_prepared` builds no map or collision provider: a configuration whose `map.mapping`
  points at files that do not exist on the pooling host evaluates fine.
- `coverage` sums across the pooled scenarios; a view with zero covered frames is `NaN`.
- Scenarios whose samples file is missing, whose sha256 does not match, or whose digests differ are
  excluded by the pooling tool and listed in its own output.
