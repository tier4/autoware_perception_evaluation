# Release note

- 上ほど新しい

## Now development for develop branch -> v1.1

- <https://github.com/tier4/AWMLevaluation/pull/31>
  - 【改良】metricsの表示でpredicted objectの総数も表示するようにする
  - 【改良】metricsの表示をmarkdown tableに変更
- <https://github.com/tier4/AWMLevaluation/pull/30>
  - 【改良】polygon と foot print をmerge
- <https://github.com/tier4/AWMLevaluation/pull/28>
  - 【改良】get_fn_objects の中の関数を分割
- <https://github.com/tier4/AWMLevaluation/pull/27>
  - 【改良】property methodを削除
- <https://github.com/tier4/AWMLevaluation/pull/24>
  - 【追加機能】Object Matching に iou 3d を追加
- <https://github.com/tier4/AWMLevaluation/pull/25>
  - 【追加機能】Object Matching に iou bev を追加
- <https://github.com/tier4/AWMLevaluation/pull/23>
  - 【追加機能】 dummy objectsを用いた，mAP, mAPHのテスト
- <https://github.com/tier4/AWMLevaluation/pull/22>
  - 【Document追加】develop branchを用いた開発プロセスをDocument化
- <https://github.com/tier4/AWMLevaluation/pull/20>
  - 【バグ修正】mAPの計算で、評価するobjectのfilterでFN objectが抜けて落ちているバグの修正
  - 【バグ修正】mAPの計算で、confidence順のsortが抜けていたので追加
  - 【機能追加】threshold_listからラベルごとのthresholdを取り出すインターフェイスを追加

## Release for main branch
### v1.0

- <https://github.com/tier4/AWMLevaluation/pull/8>
  - 【改良】mAP計算の関数共通化
  - 【追加機能】MetricsにmAPHを追加
- <https://github.com/tier4/AWMLevaluation/pull/15>
  - 【Document追加】Perceptionの再現性が低い・mAPが想定より低い問題に対しての解決法をDocument化
- <https://github.com/tier4/AWMLevaluation/pull/9>
  - 【追加機能】PassFail評価用のインターフェイス実装
  - 【追加機能】object filterにおいてx方向閾値y方向閾値を追加
  - 【追加機能】デバッグ・テスト用の objectを平行移動回転移動する関数の追加
  - 【改良】パラメータの渡し型の省略形に対応
  - 【改良】mAP計算において各labelごとにthresholdの設定をできるように
    - object filterの対応
    - printした時のmAPなどの表示も改良
- <https://github.com/tier4/AWMLevaluation/pull/7>
  - 【追加機能】Detection UseCase 評価用指標 (= Plane distance)の実装
    - <https://docs.google.com/presentation/d/1D89DUolg7Vsg_kP41kXH-mDQaoUTuoVp2FxoR8xZH6A/edit#slide=id.gf5d53bc139_32_11> 評価設計書
  - 【追加機能】mAPのmatching paramに Plane distanceを実装
  - 【追加機能】距離計算に必要な関数群を実装
- <https://github.com/tier4/AWMLevaluation/pull/5>
  - 【追加機能】Detection DataBase評価（mAP）
  - 【追加機能】Object情報・評価結果のclass設計・実装
  - 【追加機能】First prototypeとして中心間距離mAPの実装
  - 【追加機能】データセット読み込みの実装
  - 【追加機能】外部ToolとしてのAPI設計・実装
  - 【追加機能】logger・debug用の関数の実装
- <https://github.com/tier4/AWMLevaluation/pull/4>
  - 【追加機能】 ROS2 package化
- <https://github.com/tier4/AWMLevaluation/pull/2>
  - 【追加機能】poetry projectの構築
