# Release note

- 上ほど新しい

## Now development for develop branch -> v1.1.x

## Release for main branch

### v1.1.3

- <https://github.com/tier4/autoware_perception_evaluation/pull/94>

  - 【fix】整数値が入っている場合に対処するため`float`にキャスト

### v1.1.2

- <https://github.com/tier4/autoware_perception_evaluation/pull/91>

  - 【ci】`Python3.7.11`で`Poetry1.6.1`以降が使えないため、`Poetry1.5.1`に固定

- <https://github.com/tier4/autoware_perception_evaluation/pull/90>

  - 【fix】`PerceptionAnalyzer`での誤差計算時にTPが無いラベルが一つでもあれば空の`pd.DataFrame`を返すバグの修正

### v1.1.1

- <https://github.com/tier4/autoware_perception_evaluation/pull/88>

  - 【fix】速度座標系の修正

- <https://github.com/tier4/autoware_perception_evaluation/pull/87>

  - 【feat】`PerceptionAnalyzer`、`PerceptionVisualizer`の`from_scenario()`メソッドの更新

### v1.1.0

- <https://github.com/tier4/autoware_perception_evaluation/pull/70>

  - 【fix】`divide_tp_fp_objects`のtype hintingの修正

- <https://github.com/tier4/autoware_perception_evaluation/pull/81>

  - 【chore】update dependency

- <https://github.com/tier4/autoware_perception_evaluation/pull/83>

  - 【chore】update dependency

- <https://github.com/tier4/autoware_perception_evaluation/pull/7>

  - 【feat】unknownオブジェクトを評価する機能の追加

- <https://github.com/tier4/autoware_perception_evaluation/pull/78>

  - 【feat】FP再評価の機能を追加

- <https://github.com/tier4/autoware_perception_evaluation/pull/84>

  - 【chore】poetry.lockの更新

- <https://github.com/tier4/autoware_perception_evaluation/pull/79>

  - 【fix】mAP計算時にGTが0のクラス分の計算をしないように修正

### v1.0.7

- <https://github.com/tier4/autoware_perception_evaluation/pull/64>

  - 【feat】複数カメラのFrameIDに対して評価を可能にする機能の追加

- <https://github.com/tier4/autoware_perception_evaluation/pull/66>

  - 【feat】オブジェクトのフィルタ時にオリジナルのラベル名もしくはアトリビュートでフィルタする機能の追加

- <https://github.com/tier4/autoware_perception_evaluation/pull/67>

  - 【chore】semantic-pull-requestの修正

- <https://github.com/tier4/autoware_perception_evaluation/pull/63>
  - 【fix】EDAツール内でのFP results のfilterバグの修正

### v1.0.6

- <https://github.com/tier4/autoware_perception_evaluation/pull/60>

  - 【feat】traffic_lightのTrafficLightLabelクラスへの追加

- <https://github.com/tier4/autoware_perception_evaluation/pull/50>

  - 【feat】2D Analyzerの追加

- <https://github.com/tier4/autoware_perception_evaluation/pull/51>

  - 【refactor】PassFailResultのメンバ変数名の修正

- <https://github.com/tier4/autoware_perception_evaluation/pull/55>

  - 【fix】2D評価時にROIを指定しない場合にMatching Thresholdを指定した場合のバグの解消

- <https://github.com/tier4/autoware_perception_evaluation/pull/46>

  - 【fix】CenterDistance閾値の単位の[m]の削除

- <https://github.com/tier4/autoware_perception_evaluation/pull/37>
  - 【feat】2D perceptionおよびsensing可視化機能の追加3D perception可視化機能の更新

### v1.0.5

- <https://github.com/tier4/autoware_perception_evaluation/pull/58>
  - 【fix】non-detection area のクロップ時にバウンディングボックス外をクロップするように修正

### v1.0.4

- <https://github.com/tier4/autoware_perception_evaluation/pull/56>
  - 【fix】バウンディングボックスによる点群のクロップ関数の修正，評価時に non-detection area の全アノテーションボックスに対する点群を除去
- <https://github.com/tier4/autoware_perception_evaluation/pull/54>
  - 【fix】traffic light ラベル使用時の uuid のロードを instance name から Lane ID を読み込むように修正
- <https://github.com/tier4/autoware_perception_evaluation/pull/38>
  - 【feat】2D 評価用の解析ツールの追加
- <https://github.com/tier4/autoware_perception_evaluation/pull/49>
  - 【fix】タイポ修正
- <https://github.com/tier4/autoware_perception_evaluation/pull/45>
  - 【fix】f 文字列の修正

### v1.0.3

- <https://github.com/tier4/autoware_perception_evaluation/pull/41>
  - 【fix】オブジェクトマッチング時に GT の roi が`None`の場合のエラー回避
- <https://github.com/tier4/autoware_perception_evaluation/pull/42>
  - 【fix】ポリゴンに対する点群の内外判定アルゴリズムの修正
- <https://github.com/tier4/autoware_perception_evaluation/pull/43>
  - 【fix】Map 座標系での APH 計算時の RuntimeError 回避
- <https://github.com/tier4/autoware_perception_evaluation/pull/39>
  - 【chore】perception_lsim2d.py の更新
- <https://github.com/tier4/autoware_perception_evaluation/pull/12>
  - 【feat】`DynamicObject`に`FrameID`パラメータを追加
- <https://github.com/tier4/autoware_perception_evaluation/pull/36>
  - 【refactor】`FrameGroundTruth`が`raw_data`として`numpy.ndarray`を保持するように修正
- <https://github.com/tier4/autoware_perception_evaluation/pull/33>
  - 【docs】README の更新
- <https://github.com/tier4/autoware_perception_evaluation/pull/32>
  - 【fix】perception_lsim.py でのパラメータ名の修正
- <https://github.com/tier4/autoware_perception_evaluation/pull/30>
  - 【feat】未対応ラベルの追加
- <https://github.com/tier4/autoware_perception_evaluation/pull/31>
  - 【chore】pre-commit hooks の更新
- <https://github.com/tier4/autoware_perception_evaluation/pull/23>
  - 【ci】Python3.10 でのテストを追加
- <https://github.com/tier4/autoware_perception_evaluation/pull/26>
  - 【chore】pandas_profiling を pyproject.toml に追加
- <https://github.com/tier4/autoware_perception_evaluation/pull/15>
  - 【feat】Sensing 2D 評価機能の追加
- <https://github.com/tier4/autoware_perception_evaluation/pull/10>
  - 【feat】解析 API の機能追加
- <https://github.com/tier4/autoware_perception_evaluation/pull/27>
  - 【ci】CI での環境を ubuntu-20.04 に固定

### v1.0.2

- <https://github.com/tier4/autoware_perception_evaluation/pull/22>
  - 【feat】Sensing 評価時にフレーム毎で評価条件を変更できる機能の追加
- <https://github.com/tier4/autoware_perception_evaluation/pull/19>
  - 【fix】PerceptionPerformanceAnalyzer の DataFrame に正しいフレーム番号が適用されるように修正

### v1.0.1

- <https://github.com/tier4/autoware_perception_evaluation/pull/9>
  - 【docs】英語ドキュメントの追加
- <https://github.com/tier4/autoware_perception_evaluation/pull/16>
  - 【feat】Visibility クラスへの UNAVAILABLE の追加
- <https://github.com/tier4/autoware_perception_evaluation/pull/14>
  - 【refactor】ラベル名の大文字・小文字にかかわらず判定できるように修正
- <https://github.com/tier4/autoware_perception_evaluation/pull/8>
  - 【feat】Trailer ラベルのサポート

### v1.0.0

- <https://github.com/tier4/autoware_perception_evaluation/pull/3>
  - 【release】release v1.0.0
- <https://github.com/tier4/autoware_perception_evaluation/pull/2>
  - 【chore】package 名を awml_evaluation から perception_eval に変更
- <https://github.com/tier4/autoware_perception_evaluation/commit/836d1089bc5b6815486ab1033959499405e28926>
  - 【release】release
- <https://github.com/tier4/autoware_perception_evaluation/commit/6cff0094f8dc960daf241b2aff24bccf11785230>
  - 【fix】skipped フレーム数の表示
- <https://github.com/tier4/autoware_perception_evaluation/commit/bfd118c6ac47052b4ec514533c46d8cd2c7a6833>
  - 【release】release
- <https://github.com/tier4/autoware_perception_evaluation/commit/c860da917a8689c9fa7908f21c3b2b30bc191da7>
  - 【fix】visibility のロード部分のバグの修正
- <https://github.com/tier4/autoware_perception_evaluation/commit/514f49977190919914c249416ecbdec92c14aaf5>
  - 【ci】cancel previous workflows の追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/f412ae17453035ddad3a99ea7148d555ee52bf76>
  - 【docs】document 更新
- <https://github.com/tier4/autoware_perception_evaluation/commit/03fcc28fe982b373b41cabd4262181ddc9906eb2>
  - 【release】release
- <https://github.com/tier4/autoware_perception_evaluation/commit/bf52cfd396950c08829ecc45950ef1c4ed3ccbe5>
  - 【feat】アノテーションがない場合も non-detection area の評価のみ実行，オブジェクトの visibility="none"の場合は，detection_warning_results に結果を追加する機能の追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/4cecef214707ecf0dfe0349c0bcad4235c83d234>
  - 【fix】detection エリアと non-detection エリアが重なった場合に両方で同一の点群が評価されるバグの修正
- <https://github.com/tier4/autoware_perception_evaluation/commit/47a5faae8ef7104f87dd56bd076c7d2bec0a1ab9>
  - 【refactor】EvaluationManager に追加された時点でオブジェクトをフィルタするように修正
- <https://github.com/tier4/autoware_perception_evaluation/commit/7d311cc5e580a32511de0677c807e1446f9d606d>
  - 【style】get_ground_truth_now_frame()メソッドの return の type hinting に Optional を追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/cf068afc34f8d979194479d65f1e9b0b5ee19802>
  - 【release】release
- <https://github.com/tier4/autoware_perception_evaluation/commit/edab321a803c06fc93e48e927521f9be99bb645b>
  - 【feat】bbox 内点群の base_link からの最短距離にある点の算出機能の追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/20e287f26c22f33c139410538e92ece1f5c0739b>
  - 【release】release
- <https://github.com/tier4/autoware_perception_evaluation/commit/7e391b53bd0a8312ec323deb45bd65731b1b231c>
  - 【refactor】フレーム数とスキップされたフレーム番号の表示
- <https://github.com/tier4/autoware_perception_evaluation/commit/90c32b1b3c868e49ae72b7593541ce2e2c5ccbf0>
  - 【release】release
- <https://github.com/tier4/autoware_perception_evaluation/commit/82877402d4addb30529df9d981b39b12f19b880d>
  - 【feat】Perception 結果を BEV 空間で可視化する機能の追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/d481e26b316d6b8b7cd9a1a9c2a59b2a74c672d1>
  - 【fix】configure_logger()を複数回呼んだ際に log が重複しないように修正
- <https://github.com/tier4/autoware_perception_evaluation/commit/91b216ace5f13929c7b6f33968cb6dd0ea96511a>
  - 【ci】スペルチェックの追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/5ea18ca32ce74145b738846bd70013f4616e5fe2>
  - 【fix】マッチング時に 1GT に対して最大 1 予測がペアになるように修正
- <https://github.com/tier4/autoware_perception_evaluation/commit/c13344dfa5d70f1b6cd1e19b4dbd7cbf323c8bf5>
  - 【release】release
- <https://github.com/tier4/autoware_perception_evaluation/commit/2c95f0ce91e90573745cd3187e3589e75e454e0a>
  - 【feat】EDA のための可視化機能の追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/7473843534dae05a8c396c2ec34888cdbdbc7ea4>
  - 【ci】github actions で lsim mock を実行する機能の追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/1605509a245523f60e2846b56881a8da297fcd67>
  - 【fix】Python3.7.11 で math.dist()がないため修正
- <https://github.com/tier4/autoware_perception_evaluation/commit/d4bc71f21be5063c28d826898993d17ab7f8ab2a>
  - 【ci】github actions で unittest を実行する機能の追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/38d70bc20644ef9ae630be6e4d3de0f093380ca4>
  - 【release】release
- <https://github.com/tier4/autoware_perception_evaluation/commit/c6320332f72e421e92d8e8d5c514f7e481b9ed22>
  - 【feat】Detection 評価時，Ground truth の点群数に応じたフィルタ機能の追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/d1c2abc422043f04301fdbcb7234588323aaaa46>
  - 【fix】TP オブジェクトかどうかの比較を FN オブジェクト判定の追加によって，FN オブジェクトと判定されないバグを修正
- <https://github.com/tier4/autoware_perception_evaluation/commit/edae53c855a881f9b5afd5a203a2a4ef18cedd38>
  - 【test】Fail になっていた test の修正
- <https://github.com/tier4/autoware_perception_evaluation/commit/027a6b29a2934bf3d12b8081662fcbf4b94ac2fe>
  - 【feat】DynamicObject の==比較演算時に None との比較を可能にし，かつ unix time を参照する機能の追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/e9ddec6096a0e14100bef03e61eba8f6ef940940>
  - 【chore】tag が追加された際に，Release Note の draft を自動生成させるように変更
- <https://github.com/tier4/autoware_perception_evaluation/commit/9cdeaa6bac1e785d531463dd2eeda394b7e77c44>
  - 【feat】DynamicObjectWithPerceptionResult の対応ペア生成時に，同一ラベルのみとペアを組むように修正
- <https://github.com/tier4/autoware_perception_evaluation/commit/ab275960323fdc603e1ce932ebe439aa37c5a896>
  - 【docs】これまでの変更内容を document に追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/0be8e57d8a57c312869c3a6ad2eef270812e7810>
  - 【release】release
- <https://github.com/tier4/autoware_perception_evaluation/commit/fda89460d9eecb777c453bf8ed13f4ab5113228d>
  - 【feat】Tracking 評価メトリクス(CLEAR)の追加
  - 【refactor】基底クラス作成
  - 【refactor】ディレクトリ構造の変更
  - 【feat】PerceptionEvaluationConfig の引数の変更
- <https://github.com/tier4/autoware_perception_evaluation/commit/0c82d94ed5a1d18a29c61deae00859ef1d83ff8d>
  - 【feat】Plane distance マッチングの際に近傍点の選定方法の変更
  - 【feat】PassFailResult 内で TP オブジェクトの算出機能を追加
  - 【test】Unit test の追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/cc45786205377800cca392fa3f26c4e90c264539>
  - 【chore】python パッケージとして install される際の version 情報を追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/540c1ab91f7a4a897b593565c964070f322acd8a>
  - 【fix】awml_evaluation を python パッケージとして install できるように修正
- <https://github.com/tier4/autoware_perception_evaluation/commit/15b9bb0cab9e8bb13bd89b60dfaf084bc1598f7a>
  - 【chore】test 実行のための依存パッケージが poetry install されるように変更
- <https://github.com/tier4/autoware_perception_evaluation/commit/9fe5a6e487f7b2239477744e767035b12496f8ff>
  - 【chore】Issue テンプレートの追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/eff27041a79e2af4e2916a9872d7f48bf3b2a28f>
  - 【feat】lsim.py 実行時にコマンドライン引数として dataset path を指定するよう変更
- <https://github.com/tier4/autoware_perception_evaluation/commit/73a597237cdf389fad906c958ab132bde424ecf6>
  - 【fix】get_footprint 関数のバグ修正
  - 【feat】"LIDAR_CONCAT"だけでなく"LIDAR_TOP"の両方を許容するよう変更
- <https://github.com/tier4/autoware_perception_evaluation/commit/01361543157f0cf10ca4652daf42cd704583d9c7>
  - 【chore】PR テンプレートの追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/3609380ab5c3183b5ec1f19fad16a87dc79debdf>
  - 【release】release
- <https://github.com/tier4/autoware_perception_evaluation/commit/186ca0805f1b36f3830a2752e1cad7e1a7644d4a>
  - 【feat】Sensing lsim の機能部分を実装
  - 【feat】検出・被検出エリア内の点群数の算出
  - 【test】test/lsim.py に Sensing lsim 用 mock を追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/24cf3efb97b7681bcbb75116367037139874fc71>
  - 【feat】build type の ament_cmake への変更
- <https://github.com/tier4/autoware_perception_evaluation/commit/8a47589141f56f2a6c2147d4359b2796e206c5e0>
  - 【docs】document 追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/b0bc400eebd81b63d87fcd3d5b009813cc9b3318>
  - 【release】release
- <https://github.com/tier4/autoware_perception_evaluation/commit/8e606f8cdb1d079ae0a0d22d28d9e93445bda82f>
  - 【feat】autoware の linter, formatter 設定を適用
  - 【feat】Tool の version の更新
- <https://github.com/tier4/autoware_perception_evaluation/commit/f8244ff9138857830171c65f02363e5c18ed3343>
  - 【release】v1.3.0 release
- <https://github.com/tier4/autoware_perception_evaluation/commit/f5d1854ad433d0ed8468bda6bcbaad12e0b3112e>
  - 【feat】pointcloud の型を List[Tuple[float, float, float]] -> numpy.ndarray に変更
- <https://github.com/tier4/autoware_perception_evaluation/commit/3cf92e480be103e4bb43e692bb71aa1ff3e83c9e>
  - 【fix】colcon を使って install すると，install 以下にサブモジュールが存在しない状態になるのを解消
  - 【fix】pre-commit の mypy を disable
- <https://github.com/tier4/autoware_perception_evaluation/commit/34a087554c9eb90e419c9fbdde5ee61e65d1e4d4>
  - 【feat】pre-commit の追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/525b7dd0fcc85d675e5e3fa837782c716e90da8d>
  - 【test】test/metrics/test_ap.py で AP と APH のテスト
  - 【test】test/metrics/test_map.py で mAP と mAPH のテスト
- <https://github.com/tier4/autoware_perception_evaluation/commit/2c7adc01458579a90056514edf0d79d08131c24f>
  - 【feat】Dataset の load において DynamicObject の point cloud, uuid, velocity を追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/c8fe5010e8f6a75908f09d265d72376766bc06b7>
  - 【feat】Sensing lsim の result.json に残す情報を変更
  - 【docs】Sensing lsim の document を追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/8ab80bab48a197994169def7150c606d795950fd>
  - 【feat】Sensing logsim のための枠組みを実装
- <https://github.com/tier4/autoware_perception_evaluation/commit/a0089d17ed4ac10af7fd16753c6e2080d5d149c7>
  - 【feat】poetry update
- <https://github.com/tier4/autoware_perception_evaluation/commit/e05e882f7428375f3315fd424d32310c631cd0f0>
  - 【feat】poetry update
- <https://github.com/tier4/autoware_perception_evaluation/commit/e432d8c95fb104ae53c937ad1f09093001ebe7c9>
  - 【feat】get_scenario_result() -> get_scene_result() の rename
  - 【docs】分散評価に関する document 追加
  - 【release】release
- <https://github.com/tier4/autoware_perception_evaluation/commit/40f762065c6b997ae42cee84359398eb92e6d16c>
  - 【release】release
- <https://github.com/tier4/autoware_perception_evaluation/commit/39b489e927a56b093d7ac2747783414796d84ef3>
  - 【fix】 matching class の is_better_than()関数で None のハンドリングを追加
  - 【fix】get_fn_objects()関数における FN 判定で、一つでも Result があれば良い形に変更
  - 【fix】get_fn_objects()に FP object が含まれていたのを削除
  - 【fix】UseCase Fail object を FN object と TN object に分離
  - 【feat】ground truth を用いて ObjectResult を filter する機能を追加
  - 【refactor】全体的に refactoring
- <https://github.com/tier4/autoware_perception_evaluation/commit/383e9f6535d5b15300bbf3cd7e1d7b9f1dc128f7>
  - 【docs】Design document を追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/2c5a7debaa48a6b5de7b85cb6ba980907e5e86bb>
  - 【refactor】unit test の階層を整理
- <https://github.com/tier4/autoware_perception_evaluation/commit/b06c211af7e6ba7fc8420f32f2f22ff3430f741f>
  - 【feat】AP class に、matching 指標（中心間距離、面距離など）の平均・分散を追加
  - 【feat】debug 用 class 一覧 print する関数に、長い list の省略機能を追加
  - 【feat】TP metrics に mode を追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/c5cb57bb3c1a8e383badea8e57aca28ce24dca34>
  - 【docs】figure を整理
- <https://github.com/tier4/autoware_perception_evaluation/commit/a2de8d05b082a3890a188c27aee9cb39cd12fec2>
  - 【test】matching/object_matching.py の test の追加
  - 【docs】Plane distance の説明を追加
  - 【fix】Plane distance の計算におけるバグ修正
- <https://github.com/tier4/autoware_perception_evaluation/commit/0d8a7956a8ce3baede155b49d789211ab8147e52>
  - 【feat】PlaneDistanceData class に Pass fail 判断根拠に使う最近傍平面の座標点を追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/a57ba76d68d2f53de23cdd188fba1085ffb86964>
  - 【feat】Sensing logsim 用の関数の箱を用意
- <https://github.com/tier4/autoware_perception_evaluation/commit/77419117f971d4b9fdd2c4a0f186815b06d1c5c6>
  - 【refactor】object の matching する計算の meta class を object_matching.py に追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/fdfd016758aeec64df90b71e905bc9b4e5aa261a>
  - 【document 追加】README 修正
- <https://github.com/tier4/autoware_perception_evaluation/commit/afb810d5a7c540083089517b623330079f2877d4>
  - 【test】matching/objects_filter.py の test の追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/be1358fbc4ffbd4ff140c4e0cb3ac29012e3cba7>
  - 【test】common/object.py の test の追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/c7938f2a2cbcc6e60c1d39d397a9a7148229bf85>
  - 【test】common/point.py の test の追加
  - 【refactor】 common/point.py の引数におけるエラーハンドリングを追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/953c64640777c55867fa1c5c73c966ba4954d853>
  - 【feat】dataset 読み込み時に Object の uuid も読み込むように変更
- <https://github.com/tier4/autoware_perception_evaluation/commit/45666c88e7853e8a48817dfd761362fd45a49d50>
  - 【refactor】logger_configure の util 化
- <https://github.com/tier4/autoware_perception_evaluation/commit/9ff69364a857368e4ffd49ebed2176386e5ddf22>
  - 【feat】複数 scene 入力対応
- <https://github.com/tier4/autoware_perception_evaluation/commit/3b8ebddbd29a5525e41f520c51879190eecd3937>
  - 【release】release
- <https://github.com/tier4/autoware_perception_evaluation/commit/a8b40edecddca74aa3561ed073164e1c70c291a1>
  - 【refactor】dataset にある object を label ごとに個数を表示
  - 【fix】test/lsim.py において EvaluateManager が logger の反映されていなかったのを修正
- <https://github.com/tier4/autoware_perception_evaluation/commit/6e026b04f6a9a4439eac419515a445d5a269bf52>
  - 【docs】release する時の test 方法について追記
- <https://github.com/tier4/autoware_perception_evaluation/commit/d61ade7b31739e2bdfcf1fc7b5b23c403611b976>
  - 【refactor】metrics の表示で predicted object の総数も表示するようにする
  - 【refactor】metrics の表示を markdown table に変更
- <https://github.com/tier4/autoware_perception_evaluation/commit/71ea3f8da9af0e062601dfb9f39c09b2f079c396>
  - 【refactor】package.xml に nuscenes devkit dependency を追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/8e7f6a08e8cdbc9c826f87d416cdbd6628265beb>
  - 【refactor】polygon と foot print を merge
- <https://github.com/tier4/autoware_perception_evaluation/commit/72b3c800debf555611507791056ed8f9f74a073b>
  - 【refactor】get_fn_objects の中の関数を分割
- <https://github.com/tier4/autoware_perception_evaluation/commit/b258f04e98c859bb8fa156e8945860c5a46acece>
  - 【refactor】property method を削除
- <https://github.com/tier4/autoware_perception_evaluation/commit/8098caf61ed32fe09888efa441392ae9db6409c8>
  - 【feat】Object Matching に iou 3d を追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/c1a1d21ec0d43d35b002d8a5518703eb5bd21858>
  - 【feat】Object Matching に iou bev を追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/46a19263078a8553a6e409cf0f7d315ed137feb4>
  - 【feat】 dummy objects を用いた，mAP, mAPH のテスト
- <https://github.com/tier4/autoware_perception_evaluation/commit/97dd8c7bec61f3e7e565e80d7a0f13c1255dcf06>
  - 【document 追加】develop branch を用いた開発プロセスを Document 化
- <https://github.com/tier4/autoware_perception_evaluation/commit/20f6cad3078d6adc4758960bfc3b854a25c62a03>
  - 【fix】mAP の計算で、評価する object の filter で FN object が抜けて落ちているバグの修正
  - 【fix】mAP の計算で、confidence 順の sort が抜けていたので追加
  - 【feat】threshold_list からラベルごとの threshold を取り出すインターフェイスを追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/6684cf334bee0a3c6ca066c8801abeff1238057f>
  - 【refactor】mAP 計算の関数共通化
  - 【feat】Metrics に mAPH を追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/3af17c2ef36396d508ff4144ff7d986d85bfcd13>
  - 【docs】Perception の再現性が低い・mAP が想定より低い問題に対しての解決法を document 化
- <https://github.com/tier4/autoware_perception_evaluation/commit/9617c214378c741facc6119d297312dd27fd5e29>
  - 【feat】`__init__.py` の追加
- <https://github.com/tier4/autoware_perception_evaluation/commit/555e2492d9149ac82efbd9901470cf3f09bfe1b5>
  - 【feat】PassFail 評価用のインターフェイス実装
  - 【feat】object filter において x 方向閾値 y 方向閾値を追加
  - 【feat】デバッグ・テスト用の object を平行移動回転移動する関数の追加
  - 【refactor】パラメータの渡し型の省略形に対応
  - 【refactor】mAP 計算において各 label ごとに threshold の設定をできるように
    - object filter の対応
    - print した時の mAP などの表示も改良
- <https://github.com/tier4/autoware_perception_evaluation/commit/2a6fa68be840aeb42ba6a2dff2f72ec34ec1e142>
  - 【feat】Detection UseCase 評価用指標 (= Plane distance)の実装
  - 【feat】mAP の matching param に Plane distance を実装
  - 【feat】距離計算に必要な関数群を実装
  - 【feat】Detection DataBase 評価（mAP）
  - 【feat】Object 情報・評価結果の class 設計・実装
  - 【feat】First prototype として中心間距離 mAP の実装
  - 【feat】データセット読み込みの実装
  - 【feat】外部 Tool としての API 設計・実装
  - 【feat】logger・debug 用の関数の実装
- <https://github.com/tier4/autoware_perception_evaluation/commit/e8c6061221bb7566000ec8d4fce500e5dfc9b0c7>
  - 【feat】poetry project の構築
  - 【feat】 ROS2 package 化
