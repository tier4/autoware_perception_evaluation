# awml_evaluator

## Get started

- [Get started （日本語版）](docs/get_started_ja.md)
- [Release Note](docs/release_note.md)

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
  - 1 rosbag, pcd, jpeg, annotationの塊
- Frame: 1 pointcloudの入力と、その入力に対しての結果のまとまり

```
- Catalog
  - List[Scenario]
- Scenario
  - List[Scene]
- Scene
  - List[Frame]
- Frame
  - pointcloud
  - List[ground_truth_object]
  - List[predicted_object]
```

### ソフトウェアの名称

- Evaluator
  - cloud上にある評価系システムのこと
  - Lsim評価を含めた評価アプリケーションの管理が行われている
- Logsim
  - 評価を行うros packageのこと
- AWMLevaluation: Perception評価用リポジトリの名前
  - EvaluationManager: 評価の計算等を行うclass

## 開発について

- ros package化
  - 使われる時はros packageとしてincludeされる
    - autoware_utilsと同様な感じになる
  - package.xmlを追加する
  - 縛り：/awml_evaluation/awml_evaluation 以下に全てのcodeを入れる(ROSパッケージとしてimportするため)

## 未対応リスト

- [ ] BEV可視化（object filterのrefactor)
- [ ] testの追記
- [ ] dataset_path: str -> List[str] への対応
- [ ] test lsimをlsim側に移動して、ローカル読み込みのみ対応する
- [ ] T4 format v1.1対応
- [ ] 3D IoU matching
- [ ] Detection 評価
- [ ] Tracking 評価
- [ ] Prediction 評価
- [ ] Detection Use Case 評価
