# データ構造

## 点群の検知結果

1. ~Bounding box の中の点群の座標値 (List[Tuple[float]])~
   - pros: 判断根拠を残せる
   - cons: 計算量多、対して使わないのにデータ量が多くて邪魔
2. **Bounding box の中の点群の個数 (int)**
   - pros: ある程度の判断根拠を残せる、result.json は見やすい
   - cons: 計算量多
3. ~Bounding box の中の点群の有無 (bool)~
   - pros: result.json は見やすい、計算量少
   - cons: 点群の様子を見たいときに可視化が必要になる
   - awml_evaluation では 2 を採用、理由としては
     - 1.に関しては、細かい点群の位置は、viewer や rosbag で確認できれば良い
     - 現状のの Sensing lsim v1 で 2 が採用されている

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

### SensingFrameResult

1 frame に対しての検出・非検出対象エリアに対する点群のセンシング結果[(参照)](../../../awml_evaluation/awml_evaluation/evaluation/sensing/sensing_frame_result.py)

| Argument               |         type         | Description    |
| :--------------------- | :------------------: | :------------- |
| `sensing_frame_config` | `SensingFrameConfig` | config         |
| `unix_time`            |        `int`         | UNIX timestamp |
| `frame_name`           |        `str`         | フレーム番号名 |

#### 評価実行 : `SensingFrameResult.evaluate_frame()`

検出性能 / 非検出対象エリアにおける誤検出の評価を行う．

- 検出性能評価
  - `pointcloud_for_detection`の内`ground_truth_objects`の box 内に含まれる点群数は規定値以上あるかを評価する．
- 非検出対象エリアにおける誤検出評価

| Argument                       |         type          | Description                    |
| :----------------------------- | :-------------------: | :----------------------------- |
| `ground_truth_objects`         | `List[DynamicObject]` | GT オブジェクト                |
| `pointcloud_for_detection`     |    `numpy.ndarray`    | 検出性能評価のための点群       |
| `pointcloud_for_non_detection` | `List[numpy.ndarray]` | 各非検出対象エリアに対する点群 |

## Object 単位でのデータ構造

### DynamicObjectWithSensingResult

1 つの GT オブジェクト（アノテーションされた bounding box）に対しての結果[(参照)](../../../awml_evaluation/awml_evaluation/evaluation/sensing/sensing_result.py)

| Argument                       |         type          | Description                    |
| :----------------------------- | :-------------------: | :----------------------------- |
| `ground_truth_objects`         | `List[DynamicObject]` | GT オブジェクト                |
| `pointcloud_for_detection`     |    `numpy.ndarray`    | 検出性能評価のための点群       |
| `pointcloud_for_non_detection` | `List[numpy.ndarray]` | 各非検出対象エリアに対する点群 |

```txt
  Arguments:
    ground_truth_object (DynamicObject): Ground truth object
    pointcloud (numpy.ndarray): 地面除去された点群
    scale_factor (float): Bounding box のスケーリング係数

  Attributes:
    ground_truth_object (DynamicObject): 同上
    inside_pointcloud: (numpy.ndarray): Bounding box内の点群
    inside_pointcloud_num (int): Bounding box内の点群数
    is_detected (bool): Bounding box内に１つでも点があればTrue, 0ならばFalse
```
