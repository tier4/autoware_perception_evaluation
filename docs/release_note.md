# Release note

- 上ほど新しい

## Now development for develop branch -> v1.3

- <https://github.com/tier4/AWMLevaluation/pull/61>
  - 【改良】pre-commit の追加
- <https://github.com/tier4/AWMLevaluation/pull/26>
  - 【test 追加】test/metrics/test_ap.py で AP と APH のテスト
  - 【test 追加】test/metrics/test_map.py で mAP と mAPH のテスト
- <https://github.com/tier4/AWMLevaluation/pull/64/files>
  - 【機能追加】Dataset の load において DynamicObject の point cloud, uuid, velocity を追加
- <https://github.com/tier4/AWMLevaluation/pull/66>
  - 【改良】Sensing lsim の result.json に残す情報を変更
  - 【document 追加】Sensing lsim の document を追加
- <https://github.com/tier4/AWMLevaluation/pull/65>
  - 【追加】Sensing logsim のための枠組みを実装
- <https://github.com/tier4/AWMLevaluation/pull/62>
  - 【改良】poetry update
- <https://github.com/tier4/AWMLevaluation/pull/59>
  - 【改良】poetry update

## Release for main branch

### v1.2.1

- <https://github.com/tier4/AWMLevaluation/pull/57>
  - 【改良】get_scenario_result() -> get_scene_result() の rename
  - 【document 追加】分散評価に関する document 追加
  - 【release】v1.2.1 release

### v1.2

- <https://github.com/tier4/AWMLevaluation/pull/56>
  - 【release】v1.2 release
- <https://github.com/tier4/AWMLevaluation/pull/55>
  - 【バグ修正】 matching class の is_better_than()関数で None のハンドリングを追加
  - 【バグ修正】get_fn_objects()関数における FN 判定で、一つでも Result があれば良い形に変更
  - 【バグ修正】get_fn_objects()に FP object が含まれていたのを削除
  - 【バグ修正】UseCase Fail object を FN object と TN object に分離
  - 【機能追加】ground truth を用いて ObjectResult を filter する機能を追加
  - 【改良】全体的に refactoring
- <https://github.com/tier4/AWMLevaluation/pull/53>
  - 【document 追加】Design document を追加
- <https://github.com/tier4/AWMLevaluation/pull/52>
  - 【改良】unit test の階層を整理
- <https://github.com/tier4/AWMLevaluation/pull/51>
  - 【機能追加】AP class に、matching 指標（中心間距離、面距離など）の平均・分散を追加
  - 【機能追加】debug 用 class 一覧 print する関数に、長い list の省略機能を追加
  - 【機能追加】TP metrics に mode を追加
- <https://github.com/tier4/AWMLevaluation/pull/50>
  - 【document 追加】figure を整理
- <https://github.com/tier4/AWMLevaluation/pull/40>
  - 【test 追加】matching/object_matching.py の test の追加
  - 【document 追加】Plane distance の説明を追加
  - 【バグ修正】Plane distance の計算におけるバグ修正
- <https://github.com/tier4/AWMLevaluation/pull/49>
  - 【機能追加】PlaneDistanceData class に Pass fail 判断根拠に使う最近傍平面の座標点を追加
- <https://github.com/tier4/AWMLevaluation/pull/48>
  - 【機能追加】Sensing logsim 用の関数の箱を用意
- <https://github.com/tier4/AWMLevaluation/pull/44>
  - 【改良】object の matching する計算の meta class を object_matching.py に追加
- <https://github.com/tier4/AWMLevaluation/pull/47>
  - 【document 追加】README 修正
- <https://github.com/tier4/AWMLevaluation/pull/39>
  - 【test 追加】matching/objects_filter.py の test の追加
- <https://github.com/tier4/AWMLevaluation/pull/37/>
  - 【test 追加】common/object.py の test の追加
- <https://github.com/tier4/AWMLevaluation/pull/36>
  - 【test 追加】common/point.py の test の追加
  - 【改良】 common/point.py の引数におけるエラーハンドリングを追加
- <https://github.com/tier4/AWMLevaluation/pull/34>
  - 【機能追加】複数 scene 入力対応
- <https://github.com/tier4/AWMLevaluation/pull/42>
  - 【改良】logger_configure の util 化
- <https://github.com/tier4/AWMLevaluation/pull/43>
  - 【追加機能】dataset 読み込み時に Object の uuid も読み込むように変更

### v1.1

- <https://github.com/tier4/AWMLevaluation/pull/35>
  - 【release】v1.1 release
- <https://github.com/tier4/AWMLevaluation/pull/33>
  - 【改良】dataset にある object を label ごとに個数を表示
  - 【バグ修正】test/lsim.py において EvaluateManager が logger の反映されていなかったのを修正
- <https://github.com/tier4/AWMLevaluation/pull/32>
  - 【document 追加】release する時の test 方法について追記
- <https://github.com/tier4/AWMLevaluation/pull/31>
  - 【改良】metrics の表示で predicted object の総数も表示するようにする
  - 【改良】metrics の表示を markdown table に変更
- <https://github.com/tier4/AWMLevaluation/pull/30>
  - 【改良】polygon と foot print を merge
- <https://github.com/tier4/AWMLevaluation/pull/29>
  - 【改良】package.xml に nuscenes devkit dependency を追加
- <https://github.com/tier4/AWMLevaluation/pull/28>
  - 【改良】get_fn_objects の中の関数を分割
- <https://github.com/tier4/AWMLevaluation/pull/27>
  - 【改良】property method を削除
- <https://github.com/tier4/AWMLevaluation/pull/24>
  - 【機能追加】Object Matching に iou 3d を追加
- <https://github.com/tier4/AWMLevaluation/pull/25>
  - 【機能追加】Object Matching に iou bev を追加
- <https://github.com/tier4/AWMLevaluation/pull/23>
  - 【追加機能】 dummy objects を用いた，mAP, mAPH のテスト
- <https://github.com/tier4/AWMLevaluation/pull/22>
  - 【document 追加】develop branch を用いた開発プロセスを Document 化
- <https://github.com/tier4/AWMLevaluation/pull/20>
  - 【バグ修正】mAP の計算で、評価する object の filter で FN object が抜けて落ちているバグの修正
  - 【バグ修正】mAP の計算で、confidence 順の sort が抜けていたので追加
  - 【機能追加】threshold_list からラベルごとの threshold を取り出すインターフェイスを追加

### v1.0

- <https://github.com/tier4/AWMLevaluation/pull/8>
  - 【改良】mAP 計算の関数共通化
  - 【追加機能】Metrics に mAPH を追加
- <https://github.com/tier4/AWMLevaluation/pull/15>
  - 【document 追加】Perception の再現性が低い・mAP が想定より低い問題に対しての解決法を document 化
- <https://github.com/tier4/AWMLevaluation/pull/9>
  - 【追加機能】PassFail 評価用のインターフェイス実装
  - 【追加機能】object filter において x 方向閾値 y 方向閾値を追加
  - 【追加機能】デバッグ・テスト用の object を平行移動回転移動する関数の追加
  - 【改良】パラメータの渡し型の省略形に対応
  - 【改良】mAP 計算において各 label ごとに threshold の設定をできるように
    - object filter の対応
    - print した時の mAP などの表示も改良
- <https://github.com/tier4/AWMLevaluation/pull/7>
  - 【機能追加】Detection UseCase 評価用指標 (= Plane distance)の実装
    - <https://docs.google.com/presentation/d/1D89DUolg7Vsg_kP41kXH-mDQaoUTuoVp2FxoR8xZH6A/edit#slide=id.gf5d53bc139_32_11> 評価設計書
  - 【追加機能】mAP の matching param に Plane distance を実装
  - 【追加機能】距離計算に必要な関数群を実装
- <https://github.com/tier4/AWMLevaluation/pull/5>
  - 【追加機能】Detection DataBase 評価（mAP）
  - 【追加機能】Object 情報・評価結果の class 設計・実装
  - 【追加機能】First prototype として中心間距離 mAP の実装
  - 【追加機能】データセット読み込みの実装
  - 【追加機能】外部 Tool としての API 設計・実装
  - 【追加機能】logger・debug 用の関数の実装
- <https://github.com/tier4/AWMLevaluation/pull/4>
  - 【追加機能】 ROS2 package 化
- <https://github.com/tier4/AWMLevaluation/pull/2>
  - 【追加機能】poetry project の構築
