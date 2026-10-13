# 点群セマンティックセグメンテーション評価

`perception_eval` は専用のストリーミング型マネージャで点単位のセマンティックセグメンテーションを評価します。
メトリクスは `tier4/autoware-ml`（コミット `fcf86419`、PR #109）から NumPy/SciPy のみで移植したものです。
フレームはデコード済みの配列として渡すため、任意のモデル・データセットアダプタから利用できます。

## 入力契約

```python
from perception_eval.evaluation.metrics.segmentation import SegmentationFrame

frame = SegmentationFrame(
    frame_name="0",
    scene_id="db_v1/scene_uuid/0",     # lanelet マップの解決に使用。不明なら None
    coordinates=xyz,                    # (N, 3+) float, base_link
    targets=target_labels,              # (N,) int, `ignore_index` 可
    predictions=predicted_labels,       # (N,) int
    probabilities=probabilities,        # (N, C) float, 各行の和は 1
    transforms=frame_ground_truth.transforms,  # マップ系フィルタ用に (BASE_LINK, MAP) を含む TransformDict
    gt_objects=tuple(frame_ground_truth.objects),  # partial-detection 用の検出 GT ボックス
)
```

検証（`ValueError`）:

| チェック                                     | 備考                                                                       |
| -------------------------------------------- | -------------------------------------------------------------------------- |
| 形状とラベルの整数 dtype                     | 構築時に確認                                                               |
| `probabilities.shape[1] == len(class_names)` | 列数は設定と一致                                                           |
| 値が `[0, 1]` かつ有限                       | `probability_tolerance`（既定 `1e-6`）                                     |
| 各行の和が 1                                 | `probability_sum_tolerance`（既定 `1e-3`）                                 |
| `predictions == argmax(probabilities)`       | `check_argmax: true` が既定。無効化した場合も採用クラスの確率が confidence |
| `gt_objects` が `BOUNDING_BOX`               | `partial_detection` 設定時のみ                                             |

点ごとに `confidence = probabilities[prediction]`、正規化エントロピー `-sum(p log p) / log(C)`
（0 の項は 0 扱い）を導出します。`target != ignore_index`、`0 <= target < C`、`0 <= prediction < C`
を満たす点のみが _有効_ で、それ以外は全メトリクスから除外されます。

## 使い方

```python
from perception_eval.config import SegmentationEvaluationConfig
from perception_eval.manager import SegmentationEvaluationManager

config = SegmentationEvaluationConfig(
    dataset_paths=[],
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
for frame in frames:                 # SegmentationFrame を 1 フレームずつ
    summary = manager.add_frame(frame)
report = manager.get_scene_result(save_report=True)   # <log_directory>/segmentation_metrics.json を出力
print(report)
```

`manager.frame_from_ground_truth(ground_truth, targets, predictions, probabilities)` は読み込んだ
`FrameGroundTruth` からフレームを構築します（座標は既定で `LIDAR_CONCAT`/`LIDAR_TOP` 点群。
この経路では `load_raw_data=True` と `load_ground_truth=True` が必要）。

`python -m test.segmentation_lsim --use_tmpdir` は合成フレームで全体を実行するモックです。

## 設定キー

| キー                        | 既定値 | 意味                                                                               |
| --------------------------- | ------ | ---------------------------------------------------------------------------------- |
| `class_names`               | 必須   | 確率列の順序に対応する学習クラス                                                   |
| `ignore_index`              | `-1`   | 全メトリクスから除外する target ラベル                                             |
| `ranges`                    | `[]`   | 点 / ボックス中心の BEV 距離窓 `[min, max)`                                        |
| `class_groups`              | `null` | `class_names` の完全分割。`grouped` ビューを追加                                   |
| `filters`                   | `[]`   | `corridor`（マップ不要）、`region`、`collision`（`map` 必須）                      |
| `components`                | `[]`   | 閉じたレジストリ（下表）                                                           |
| `map`                       | `null` | `{resolver: t4_scene_directory, data_root}` または `{resolver: explicit, mapping}` |
| `box_label_to_seg_class`    | `{}`   | `AutowareLabel` 値 -> ボックス内の点が持つべきセグメンテーションクラス             |
| `check_argmax`              | `true` | `predictions == argmax(probabilities)` を要求                                      |
| `probability_tolerance`     | `1e-6` | `[0, 1]` からの許容超過                                                            |
| `probability_sum_tolerance` | `1e-3` | 行和の 1 からの許容偏差                                                            |
| `include_confusion_cells`   | `true` | `confusion_<true>__<pred>` キーを出力                                              |

未知のキー、未知のコンポーネント種別/パラメータ、名前の重複、分割になっていないクラスグループ、
未対応ラベルはいずれも設定時点で失敗します（`MetricsConfigError`）。

## メトリクスキー

全ての値はスラッシュ区切りのキーで出力されます。

```text
segmentation/<taxonomy?>/<filter?>/<range?>/<metric-key>
```

taxonomy 階層はグループビュー（`grouped`）のみ、filter 階層は identity 以外のフィルタのみ、
range 階層（`0m_30m`、`30m_inf` など）はレンジ設定時のみ現れます。
例: `segmentation/mIoU`、`segmentation/0m_30m/error_rate`、
`segmentation/grouped/region_road/error_clusters_per_frame`。`report.to_flat_keys()` でフラット形式
（`segmentation_grouped_region_road_error_clusters_per_frame`）も得られます。

| コンポーネント（`type`）       | パラメータ（既定値）                         | キー                                                                                                                                                                     |
| ------------------------------ | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `confusion_matrix`             | -                                            | `confusion_<true>__<pred>` 点数の生カウント                                                                                                                              |
| `iou`                          | -                                            | `mIoU`（GT を持つクラスのマクロ平均）、`fwIoU`、`iou_<c>`                                                                                                                |
| `accuracy`                     | -                                            | `acc`（正解点 / 有効点）                                                                                                                                                 |
| `precision_recall_f1`          | -                                            | `mRecall`、`mPrecision`、`mF1`、`recall_<c>`、`precision_<c>`、`f1_<c>`                                                                                                  |
| `calibration`（E1）            | `num_bins=15`                                | `ece`（全点）、`ece_macro`（予測クラス別 ECE の平均）                                                                                                                    |
| `uncertainty_usefulness`（E2） | `num_bins=8192`                              | `entropy_auroc`（2 つのヒストグラムからの tie-aware AUROC）、`mean_entropy_wrong`、`mean_entropy_correct`                                                                |
| `confident_error`（E3）        | `entropy_threshold=0.3`                      | `confident_error_rate` = エントロピー `< threshold` の誤り点 / 誤り点、`confident_error_count`                                                                           |
| `error_clusters`（D1）         | `cluster_radius=0.5`、`min_cluster_points=1` | `error_rate`、`error_cluster_count`、`error_clusters_per_frame`、`error_rate_<c>`、`error_cluster_count_<c>`（true クラス別）                                            |
| `tolerant_error`（D3）         | `radius=0.2`                                 | `tolerant_error_rate`、`tolerant_error_count`、クラス別。`radius` 内に真のクラスを予測した点があれば誤りを免除。`radius=0` は厳密誤り率に一致                            |
| `partial_detection`（D2）      | `half_saturation=1.0`、`min_points=1`        | `pd_score_<det class>`、`mpd_score`、`pd_skipped_low_point_boxes`。yaw を考慮した BEV フットプリント内の `n` 点中 `k` 正解で `(k/(k+h)) / (n/(n+h))`。raw タクソノミのみ |

NaN の規則: 分母が 0 の比率は `NaN`（`0` にはしない）、カウントは `0.0`。クラス別の
IoU/precision/recall は GT を持つクラスのみ出力します。グループビューでは確率を先に畳み込んでから
confidence とエントロピーを導出するため、グループの confidence は予測グループの確率質量の和です。

## マップ依存フィルタのカバレッジ

region / collision フィルタには自車姿勢（`(BASE_LINK, MAP)` 変換）と `scene_id` に対応する
lanelet マップが必要です。どちらかを欠くフレームはそのフィルタのビューからのみ除外され、レポートは
`coverage[filter] = (covered_frames, seen_frames)` と、部分的にしかカバーされないフィルタごとに
1 件の警告を持ちます。1 フレームもカバーしなかったフィルタの値は全て `NaN` になります（誤解を招く 0 にはなりません）。

## メモリ保証

マネージャはフレームを保持しません。混同行列は固定サイズの `(F+1, R+1, C, C)` 整数配列、
キャリブレーション・エントロピー統計は固定サイズのビン、クラスタ / 近傍メトリクスはフレームごとに計算して
カウンタへ縮約します。ピークメモリはシーン長ではなく最大の 1 フレームで決まります
（`test_manager.py::test_memory_stays_bounded` 参照）。
