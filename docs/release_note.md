# Release note

- 上ほど新しい

## Now development for develop branch -> v1.3

- <https://github.com/tier4/AWMLevaluation/pull/26>
  - 【test追加】test/metrics/test_ap.py でAPとAPHのテスト
  - 【test追加】test/metrics/test_map.py でmAPとmAPHのテスト
- <https://github.com/tier4/AWMLevaluation/pull/64/files>
  - 【機能追加】Dataset のloadにおいて DynamicObject の point cloud, uuid, velocity を追加
- <https://github.com/tier4/AWMLevaluation/pull/66>
  - 【改良】Sensing lsimのresult.jsonに残す情報を変更
  - 【document追加】Sensing lsimのdocumentを追加
- <https://github.com/tier4/AWMLevaluation/pull/65>
  - 【追加】Sensing logsimのための枠組みを実装
- <https://github.com/tier4/AWMLevaluation/pull/62>
  - 【改良】poetry update
- <https://github.com/tier4/AWMLevaluation/pull/59>
  - 【改良】poetry update

## Release for main branch
### v1.2.1

- <https://github.com/tier4/AWMLevaluation/pull/57>
  - 【改良】get_scenario_result() -> get_scene_result() のrename
  - 【document追加】分散評価に関するdocument追加
  - 【release】v1.2.1 release

### v1.2

- <https://github.com/tier4/AWMLevaluation/pull/56>
  - 【release】v1.2 release
- <https://github.com/tier4/AWMLevaluation/pull/55>
  - 【バグ修正】 matching class のis_better_than()関数でNoneのハンドリングを追加
  - 【バグ修正】get_fn_objects()関数におけるFN判定で、一つでもResultがあれば良い形に変更
  - 【バグ修正】get_fn_objects()にFP objectが含まれていたのを削除
  - 【バグ修正】UseCase Fail objectをFN objectとTN objectに分離
  - 【機能追加】ground truthを用いてObjectResultをfilterする機能を追加
  - 【改良】全体的にrefactoring
- <https://github.com/tier4/AWMLevaluation/pull/53>
  - 【document追加】Design documentを追加
- <https://github.com/tier4/AWMLevaluation/pull/52>
  - 【改良】unit testの階層を整理
- <https://github.com/tier4/AWMLevaluation/pull/51>
  - 【機能追加】AP classに、matching指標（中心間距離、面距離など）の平均・分散を追加
  - 【機能追加】debug用class一覧printする関数に、長いlistの省略機能を追加
  - 【機能追加】TP metricsにmodeを追加
- <https://github.com/tier4/AWMLevaluation/pull/50>
  - 【document追加】figureを整理
- <https://github.com/tier4/AWMLevaluation/pull/40>
  - 【test追加】matching/object_matching.py のtestの追加
  - 【document追加】Plane distance の説明を追加
  - 【バグ修正】Plane distance の計算におけるバグ修正
- <https://github.com/tier4/AWMLevaluation/pull/49>
  - 【機能追加】PlaneDistanceData classに Pass fail判断根拠に使う最近傍平面の座標点を追加
- <https://github.com/tier4/AWMLevaluation/pull/48>
  - 【機能追加】Sensing logsim用の関数の箱を用意
- <https://github.com/tier4/AWMLevaluation/pull/44>
  - 【改良】objectのmatchingする計算のmeta classを object_matching.py に追加
- <https://github.com/tier4/AWMLevaluation/pull/47>
  - 【document追加】README修正
- <https://github.com/tier4/AWMLevaluation/pull/39>
  - 【test追加】matching/objects_filter.py のtestの追加
- <https://github.com/tier4/AWMLevaluation/pull/37/>
  - 【test追加】common/object.py のtestの追加
- <https://github.com/tier4/AWMLevaluation/pull/36>
  - 【test追加】common/point.py のtestの追加
  - 【改良】 common/point.py の引数におけるエラーハンドリングを追加
- <https://github.com/tier4/AWMLevaluation/pull/34>
  - 【機能追加】複数scene入力対応
- <https://github.com/tier4/AWMLevaluation/pull/42>
  - 【改良】logger_configureのutil化
- <https://github.com/tier4/AWMLevaluation/pull/43>
  - 【追加機能】dataset読み込み時にObjectのuuidも読み込むように変更

### v1.1

- <https://github.com/tier4/AWMLevaluation/pull/35>
  - 【release】v1.1 release
- <https://github.com/tier4/AWMLevaluation/pull/33>
  - 【改良】datasetにあるobjectをlabelごとに個数を表示
  - 【バグ修正】test/lsim.pyにおいてEvaluateManagerがloggerの反映されていなかったのを修正
- <https://github.com/tier4/AWMLevaluation/pull/32>
  - 【document追加】releaseする時のtest方法について追記
- <https://github.com/tier4/AWMLevaluation/pull/31>
  - 【改良】metricsの表示でpredicted objectの総数も表示するようにする
  - 【改良】metricsの表示をmarkdown tableに変更
- <https://github.com/tier4/AWMLevaluation/pull/30>
  - 【改良】polygon と foot print をmerge
- <https://github.com/tier4/AWMLevaluation/pull/29>
  - 【改良】package.xml にnuscenes devkit dependencyを追加
- <https://github.com/tier4/AWMLevaluation/pull/28>
  - 【改良】get_fn_objects の中の関数を分割
- <https://github.com/tier4/AWMLevaluation/pull/27>
  - 【改良】property methodを削除
- <https://github.com/tier4/AWMLevaluation/pull/24>
  - 【機能追加】Object Matching に iou 3d を追加
- <https://github.com/tier4/AWMLevaluation/pull/25>
  - 【機能追加】Object Matching に iou bev を追加
- <https://github.com/tier4/AWMLevaluation/pull/23>
  - 【test追加】 dummy objectsを用いた，mAP, mAPHのテスト
- <https://github.com/tier4/AWMLevaluation/pull/22>
  - 【document追加】develop branchを用いた開発プロセスをDocument化
- <https://github.com/tier4/AWMLevaluation/pull/20>
  - 【バグ修正】mAPの計算で、評価するobjectのfilterでFN objectが抜けて落ちているバグの修正
  - 【バグ修正】mAPの計算で、confidence順のsortが抜けていたので追加
  - 【機能追加】threshold_listからラベルごとのthresholdを取り出すインターフェイスを追加

### v1.0

- <https://github.com/tier4/AWMLevaluation/pull/8>
  - 【改良】mAP計算の関数共通化
  - 【機能追加】MetricsにmAPHを追加
- <https://github.com/tier4/AWMLevaluation/pull/15>
  - 【document追加】Perceptionの再現性が低い・mAPが想定より低い問題に対しての解決法をdocument化
- <https://github.com/tier4/AWMLevaluation/pull/9>
  - 【機能追加】PassFail評価用のインターフェイス実装
  - 【機能追加】object filterにおいてx方向閾値y方向閾値を追加
  - 【機能追加】デバッグ・テスト用の objectを平行移動回転移動する関数の追加
  - 【改良】パラメータの渡し型の省略形に対応
  - 【改良】mAP計算において各labelごとにthresholdの設定をできるように
    - object filterの対応
    - printした時のmAPなどの表示も改良
- <https://github.com/tier4/AWMLevaluation/pull/7>
  - 【機能追加】Detection UseCase 評価用指標 (= Plane distance)の実装
    - <https://docs.google.com/presentation/d/1D89DUolg7Vsg_kP41kXH-mDQaoUTuoVp2FxoR8xZH6A/edit#slide=id.gf5d53bc139_32_11> 評価設計書
  - 【機能追加】mAPのmatching paramに Plane distanceを実装
  - 【機能追加】距離計算に必要な関数群を実装
- <https://github.com/tier4/AWMLevaluation/pull/5>
  - 【機能追加】Detection DataBase評価（mAP）
  - 【機能追加】Object情報・評価結果のclass設計・実装
  - 【機能追加】First prototypeとして中心間距離mAPの実装
  - 【機能追加】データセット読み込みの実装
  - 【機能追加】外部ToolとしてのAPI設計・実装
  - 【機能追加】logger・debug用の関数の実装
- <https://github.com/tier4/AWMLevaluation/pull/4>
  - 【機能追加】 ROS2 package化
- <https://github.com/tier4/AWMLevaluation/pull/2>
  - 【機能追加】poetry projectの構築
