## 用語定義

### データ構造の用語定義 (TBD)

- Catalog: Scenarioの塊
  - 例：右折UseCase = (UC-0001, UC-0003, UC-0006)
- Scenario: 評価する単位
  - Scenarioには2つ種類がある
  - DB-Scenario: DataBase用Datasetで主に学習用に用いる
    - 1 Scenario = DB-Scenario, n scene
  - UC-Scenario: UseCase評価用Dataset
    - 1 Scenario = UC-Scenario, 1 scene
- Scene: 1 連続 rosbagの単位
  - 1 rosbagから構成されるデータ群
  - 1 rosbag + pcd + jpeg + annotationの塊
- FrameResult: 1 pointcloudの入力と、その入力に対しての結果のまとまり
  - 1 pointcloud + n ground truth objects + n object results (List[DynamicObjectWithResult])
- DynamicObjectWithResult: 1 predicted object（認識の推論結果のbounding box）に対しての結果
  - ground_truth_object: Ground truth
  - predicted_object: (Autowareの) 認識の推論結果
  - metrics_score: 評価数値

```
- Catalog
  - List[Scenario]
- Scenario
  - List[Scene]
- Scene
  - List[FrameResult]
- FrameResult
  - pointcloud
  - List[ground_truth_object]
  - List[DynamicObjectWithResult]
- DynamicObjectWithResult
  - ground_truth_object
  - predicted_object
  - metrics_score

```

### ソフトウェアの名称

- Evaluator
  - cloud上にある評価系システムのこと
  - Lsim評価を含めた評価アプリケーションの管理が行われている
- [logsim](https://github.com/tier4/logsim)
  - 評価を行うros packageのこと
  - [Perception logsim](https://github.com/tier4/logsim/blob/ros2/logsim/scripts/perception_evaluator_node.py)
- AWMLevaluation: Perception評価用リポジトリの名前
  - EvaluationManager: 評価の計算等を行うclass

## 評価について

| Autoware module                   | 座標系         |
| :-------------------------------- | :------------- |
| detection/objects                 | baselink座標系 |
| tracking/objects                  | map座標系      |
| objects (-> prediction/objects ?) | map座標系      |
| pointcloud                        | baselink座標系 |
