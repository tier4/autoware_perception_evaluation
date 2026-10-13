# Advanced detection metrics: ファイル仕様（schema v1）

運転文脈を考慮した検出メトリクス（[metrics.md](metrics.md)）は
シナリオ毎に計算し、その後シナリオ横断でプールします。それを可能にするのがシナリオ毎の 2 つのファイル、
**レポート**（`advanced_detection_metrics.json`）と **prepared samples**（`advanced_detection_samples.npz`）です。
samples ファイルの形式（`PreparedSamples.to_npz` / `from_npz`）と、レポート内の `config_digest`, `values`,
`coverage`, `counts`, `warnings` は `perception_eval` が定義します。レポートファイル自体は下流のログ再生ツール
（例: `driving_log_replayer_v2`）が書き出し、producer やシーンの情報を付加します。プーリングツールは両ファイルを
読み戻して `evaluate_prepared(config, parts)` を呼びます。

両ファイルは `scene_result.pkl` の隣の `result_archive/<topic.with.dots>/` にアトミックに書き込まれます
（`<name>.tmp` に書いてから `os.replace`）。`advanced_detection_metrics` が設定され、かつ評価タスクが
`fp_validation` でない場合にのみ書き出されます。対象は無視フレームを除去した**後**のフレームで、
suite の実行は**1 回**（`prepare` 1 回、`evaluate_prepared` 1 回）です。

## バージョンと digest

- `PREPARED_SCHEMA_VERSION = 1`（`perception_eval.evaluation.metrics.detection.prepared`）。
  それ以外の `schema_version` は `PoolingError` で拒否されます。
- `config_digest(config)` = 以下の正規化 JSON（`sort_keys=True`, `separators=(",", ":")`）の `sha256:<hex>`:
  `{"class_names": [...], "advanced": <config.serialization() から map.data_root と map.mapping を除いたもの>, "prepared_schema_version": 1}`。
  マップの置き場所と dict の順序には依存せず、コンポーネント・フィルタ・レンジ・クラスグループ・マッチング・score の
  いずれかのパラメータが変わると変化します。
- `frame_filter_digest`（レポートのみ）= `evaluation_config_dict` のキー `target_labels`, `max_x_position`,
  `max_y_position`, `max_distance`, `min_distance`, `min_point_numbers`, `confidence_threshold`, `label_prefix`,
  `merge_similar_labels`, `allow_matching_unknown`（無いキーは `null`）の正規化 JSON の sha256。
  フレームフィルタが異なるシナリオは異なる object 集合を評価しているため、一緒にはプールしません。

## `advanced_detection_samples.npz`

`np.savez_compressed` で書き出し、object 配列を含まず、`np.load(path, allow_pickle=False)` で読みます。
`F` フレーム、`P` 予測、`G` 正解、`K = len(filter_names)`。`F = 0`（空のシーン）も有効です。

| key                          | dtype / shape                    | 備考                                                                                                                                                                                                                                                    |
| ---------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `header`                     | `uint8[N]`                       | UTF-8 JSON: `{"schema_version": 1, "config_digest": "sha256:...", "class_names": [...], "filter_names": [...], "has_ttc": bool, "box_columns": ["cx","cy","cz","dx","dy","dz","yaw","vx","vy"], "counts": {...}, "warnings": [...], "producer": {...}}` |
| `frame_name`                 | `<U64 [F]`                       | 入力順のフレーム名（スキップされたフレームは含まない）                                                                                                                                                                                                  |
| `unix_time`                  | `int64[F]`                       |                                                                                                                                                                                                                                                         |
| `ttc_covered`                | `bool[F]`                        | シーンに lanelet マップが無い、またはフレームに自車姿勢が無い場合 False                                                                                                                                                                                 |
| `available`                  | `bool[F, K]`                     | フレーム × 非 identity フィルタ（列順 = `filter_names`）                                                                                                                                                                                                |
| `pred_offsets`, `gt_offsets` | `int64[F + 1]`                   | CSR の行オフセット: フレーム `i` の行は `offsets[i]:offsets[i + 1]`                                                                                                                                                                                     |
| `pred_boxes`, `gt_boxes`     | `float64[P, 9]`, `float64[G, 9]` | base_link の行 `[cx, cy, cz, dx(length), dy(width), dz(height), yaw, vx, vy]`                                                                                                                                                                           |
| `pred_scores`                | `float64[P]`                     | 設定した `score_source`。非有限の score は除去済み                                                                                                                                                                                                      |
| `pred_labels`, `gt_labels`   | `int64[P]`, `int64[G]`           | `class_names` へのインデックス                                                                                                                                                                                                                          |
| `pred_ttc`, `gt_ttc`         | `float64[P]`, `float64[G]`       | `has_ttc` の場合のみ。到達可能性 TTC [s]、`inf` = 到達不能                                                                                                                                                                                              |
| `pred_masks`, `gt_masks`     | `bool[P, K]`, `bool[G, K]`       | 各フィルタが残す要素。フィルタが利用不能なフレームでは全て False                                                                                                                                                                                        |

`counts` は `input_frames`, `skipped_frames`, `polygon_as_box`, `non_finite_score`, `pred_total`, `pred_kept`,
`zero_score` を持ちます。`producer` は自由形式（`{"name": ..., "version": ...}`）で、未指定なら `{}` です。
`to_npz` は書き出したファイルの sha256（hex）を返し、レポートがそれを記録します。

## `advanced_detection_metrics.json`（シナリオ毎）

厳密な JSON（`json.dump(..., allow_nan=False)`）で、`NaN` は全て `null` として書かれます。

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

| フィールド                             | 意味                                                                                                                                                             |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema`, `schema_version`             | `perception_eval.advanced_detection_report`, `1`                                                                                                                 |
| `status`, `error`                      | `ok`。準備後にフレームが残らなければ `empty`。suite が例外を出した場合は `error` でメッセージを `error` に入れる（`values` は `{}`、samples ファイルは書かない） |
| `producer`                             | ファイルを書いたツールとそのバージョン                                                                                                                           |
| `perception_eval`                      | 本パッケージの `version` と samples ファイルの `prepared_schema_version`                                                                                         |
| `topic`, `evaluation_task`, `frame_id` | 評価した topic、タスク（`detection`/`tracking`）、object の frame id                                                                                             |
| `scene`                                | `dataset_name`（T4 dataset パスの basename）と lanelet マップが解決できたか                                                                                      |
| `config_digest`, `frame_filter_digest` | 上記参照。プーリングツールは両方でシナリオをグループ化する                                                                                                       |
| `config`                               | `AdvancedDetectionMetricsConfig.serialization()` から `map.data_root` / `map.mapping` を除き、`class_names` を加えたもの                                         |
| `frames`                               | 受け取った `frame_results`、`ignored`（評価前に除去）、`evaluated`（= `prepare` に渡した数）、suite が `skipped` した数、`prepared`（= `evaluated - skipped`）   |
| `counts`                               | `PreparedSamples.counts`（同じキー）                                                                                                                             |
| `coverage`                             | `MetricReport.coverage` を `[covered, seen]` の組で                                                                                                              |
| `values`                               | このシナリオ単体の `MetricReport.values`（`null` = NaN）。プール後の値は samples から再計算し、これらを平均することはない                                        |
| `warnings`                             | `MetricReport.warnings`                                                                                                                                          |
| `samples_file`                         | レポートの隣にある samples ファイルの名前、sha256、`num_frames`                                                                                                  |

## プーリング

```python
from perception_eval.evaluation.metrics.detection.prepared import PreparedSamples, evaluate_prepared

parts = [PreparedSamples.from_npz(path) for path in paths]  # 正規の順序（下記）
report = evaluate_prepared(config, parts)  # digest / class / filter が合わなければ PoolingError
```

- 正規のプーリング順序: シナリオを `(dataset_name, scenario_path)` でソートし、フレームはファイル順。
  この順序なら `evaluate_prepared(parts)` は連結したフレームに対する 1 回の `suite.evaluate` とビット単位で一致します。
  同一 score は入力順にマッチされる（安定ソート）ため、parts の順序が違うと同点の扱いが変わり得ますが、
  どのフレームがプールされるかは変わりません。
- `evaluate_prepared` はマップや collision provider を生成しません。プーリングホストに存在しないファイルを
  `map.mapping` が指していても問題なく評価できます。
- `coverage` はプールしたシナリオ全体で合計され、カバーされたフレームが 0 のビューは `NaN` です。
- samples ファイルが無い、sha256 が一致しない、digest が異なるシナリオはプーリングツールが除外し、
  その出力に一覧します。
