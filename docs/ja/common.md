# Common items in Perception and Sensing

## [`<class> DynamicObject(...)`](../../perception_eval/perception_eval/common/object/object3d.py)

3 次元オブジェクトのクラス．

- 使用される評価タスク: `DETECTION`, `TRACKING`, `PREDICTION`, `SENSING`

| Argument                 |                     type                     | Description                                                             |
| :----------------------- | :------------------------------------------: | :---------------------------------------------------------------------- |
| `unix_time`              |                    `int`                     | Unix time .                                                             |
| `frame_id`               |                  `FrameID`                   | 3 次元オブジェクトが従う FrameID インスタンス．BASE_LINK または MAP．   |
| `position`               |         `Tuple[float, float, float]`         | 位置 (x, y, z).                                                         |
| `orientation`            |                 `Quaternion`                 | クォータニオン (w, x, y, z).                                            |
| `shape`                  |                   `Shape`                    | Shapeインスタンス．ボックスサイズ (width, length, depth)の情報等を持つ. |
| `velocity`               |         `Tuple[float, float, float]`         | 速度 (vx, vy, vz).                                                      |
| `semantic_score`         |                   `float`                    | 信頼度 [0, 1].                                                          |
| `semantic_label`         |                 `LabelType`                  | ラベル名.                                                               |
| `pointcloud`             |          `Optional[numpy.ndarray]`           | 点群. (Default: None)                                                   |
| `uuid`                   |               `Optional[str]`                | オブジェクトの UUID. (Default: None)                                    |
| `tracked_positions`      | `Optional[List[Tuple[float, float, float]]]` | 過去からの位置遷移のリスト. (Default: None)                             |
| `tracked_orientations`   |            `Optional[Quaternion]`            | 過去からのクオータニオン遷移のリスト. (Default: None)                   |
| `tracked_sizes`          | `Optional[List[Tuple[float, float, float]]]` | 過去からのボックスサイズの遷移のリスト. (Default: None)                 |
| `tracked_twists`         | `Optional[List[Tuple[float, float, float]]]` | 過去からの速度遷移のリスト. (Default: None)                             |
| `predicted_positions`    | `Optional[List[Tuple[float, float, float]]]` | 予測された将来の位置の遷移のリスト. (Default: None)                     |
| `predicted_orientations` |            `Optional[Quaternion]`            | 予測された将来のクォータニオンの遷移のリスト. (Default: None)           |
| `predicted_sizes`        | `Optional[List[Tuple[float, float, float]]]` | 予測された将来のボックスサイズの遷移のリスト. (Default: None)           |
| `predicted_twists`       | `Optional[List[Tuple[float, float, float]]]` | 予測された将来の速度遷移のリスト. (Default: None)                       |
| `predicted_confidence`   |              `Optional[float]`               | 予測状態の信頼度. (Default: None)                                       |
| `visibility`             |            `Optional[Visibility]`            | 視認性のステータス. (Default: None)                                     |

## [`<class> DynamicObject2D(...)`](../../perception_eval/perception_eval/common/object/object2d.py)

2 次元オブジェクトのクラス．

- 使用される評価タスク: `DETECTION2D`, `TRACING2D`, `CLASSIFICATION2D`

| Argument         |                 type                  | Description                                                |
| :--------------- | :-----------------------------------: | :--------------------------------------------------------- |
| `unix_time`      |                 `int`                 | Unix time .                                                |
| `frame_id`       |               `FrameID`               | 2 次元オブジェクトが従う FrameID インスタンス．CAM\_\*\*． |
| `semantic_score` |                `float`                | 信頼度 [0, 1].                                             |
| `semantic_label` |              `LabelType`              | ラベル名.                                                  |
| `roi`            | `Optional[Tuple[int, int, int, int]]` | (x_min, y_min, height, width) of ROI. (Default: None)      |
| `uuid`           |            `Optional[str]`            | オブジェクトの UUID. (Default: None)                       |
| `visibility`     |        `Optional[Visibility]`         | 視認性のステータス. (Default: None)                        |

## Ground truth

### [`<class> FrameGroundTruth(...)`](../../perception_eval/perception_eval/common/dataset/ground_truth.py)

フレームごとの GT オブジェクトの集合のクラス．

| Argument     |           type            | Description                                                                    |
| :----------- | :-----------------------: | :----------------------------------------------------------------------------- |
| `unix_time`  |           `int`           | Unix time.                                                                     |
| `frame_name` |           `str`           | フレーム名.                                                                    |
| `objects`    |    `List[ObjectType]`     | GT オブジェクトのリスト.                                                       |
| `ego2map`    | `Optional[numpy.ndarray]` | オブジェクトの座標系を base_link から map に変換する 4x4 行列. (Default: None) |
| `raw_data`   | `Optional[numpy.ndarray]` | 点群または画像. (Default: None)                                                |

### [`<func> load_all_datasets(...) -> List[FrameGroundTruth]`](../../perception_eval/perception_eval/common/dataset/load.py)

データセットをロードする関数．

| Argument          |                type                 | Description                                     |
| :---------------- | :---------------------------------: | :---------------------------------------------- |
| `dataset_paths`   |             `List[str]`             | データセットのパス.                             |
| `evaluation_task` |          `EvaluationTask`           | 評価タスク名.                                   |
| `label_converter` |          `LabelConverter`           | LabelConverter のインスタンス.                  |
| `frame_id`        | `Union[FrameID, Sequence[FrameID]]` | オブジェクトが従う FrameID インスタンス．       |
| `load_raw_data`   |               `bool`                | 点群/画像をロードするかどうか. (Default: False) |
