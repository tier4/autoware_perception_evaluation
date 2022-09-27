# Perception performance analyzer

Perception の評価結果をもとに解析を行う．

## Perception 評価種別

評価・解析方法として，UseCase/DataBase の 2 種類がある．

- UseCase 評価 = 設計されたテストケースにおいて，対象物体に着目してメトリクスの評価・PASS/FAIL の判定を行う

  - 認識性能の精度部分を評価，安全性能の担保をメインの目的とした評価

- DataBase = 実環境で収集したデータに対して，視野内の物体に対してメトリクスの評価を行う

  - Perception アルゴリズム性能の相対評価に利用，安全担保の基準には基本的にはならない

## 解析内容

1. TP 率
2. FN 率
3. FP 率
4. TP 時の位置精度
   - 自車座標系の縦方向/横方向にそれぞれ分けて誤差の RMS, 分散, 最大/最小値を算出
5. TP 時の Heading 精度(4.と同じ)
6. TP 時の速度精度(4.と同じ, tracking のみ)
7. 各クラスの性能(AP, MOTA 等)

- UseCase

  - 一つの物体を対象として解析を行うことを前提としているため，評価時にはシナリオや PerceptionEvaluationConfig にオブジェクトを絞るような設定をする．
  - 上記の解析内容に加えて，対象物体の GT との位置誤差を時系列ごとにプロットする

- DataBase
  - 複数のデータセットに対する評価結果をまとめて解析することが可能
  - 上記の解析内容に加えて，評価対象領域を 1, 3, 9 のいずれかで分割し，それぞれの領域内での解析・GT/予測物体の数をプロットする．

## How to use

driving_log_replayer を使用する際にはシナリオ(`.yaml`)から config が生成され，評価結果(`List[PerceptionFrameResult]`)が`.pkl`で保存される．

### 1. Initialization

初期化方法には，1. `PerceptionEvaluationConfig`から，2. シナリオファイル(`.yaml`)からの 2 パターンがある．

```python
from perception_eval.tool.perception_performance_analyzer import PerceptionPerformanceAnalyzer


# 1. with PerceptionEvaluationConfig
# REQUIRED:
#   - evaluation_config <PerceptionEvaluationConfig>

analyzer = PerceptionPerformanceAnalyzer(evaluation_config)

# 2. with scenario file (.yaml)
# REQUIRED:
#   - scenario_path <str>

analyzer = PerceptionPerformanceAnalyzer.from_scenario(
    result_root_directory,
    scenario_path,
)
```

### 2. Add frame results

`PerceptionFrameResult`の追加方法には，1. `PerceptionFrameResult`を直接入力，2. `pickle`ファイルのパスを入力する 2 パターンがある．

```python
# 1. with PerceptionFrameResult
# REQUIRED:
#   - frame_results <List[PerceptionFrameResult]>
analyzer.add(frame_results)


# 2. with pickle file (.pkl)
# REQUIRED:
#   - pickle_path: <str>
analyzer.add_from_pkl(pickle_path)
```

```python
>>> analyzer.df
                    timestamp          x          y         w          l         h       yaw            vx        vy                      nn_point1                      nn_point2 label                              uuid status  area  frame  scene
0  ground_truth  1.603763e+15  85.536254   2.151734  3.237000  12.112000  3.816000  0.017880  1.302359e-02  0.044080  (-7.172930, -49.5320, -0.2938) (-7.172930, -49.5320, -0.2938) truck  a0e19d9fc8e528fb471d0d29bdf32927     TP   0.0   48.0    1.0
   estimation    1.603763e+15  83.445015   2.306474  2.821630   6.807208  2.983142  0.030410  4.937477e-09  0.000000  truck                              None     TP   0.0   48.0    1.0
...
```

### 3. Analyze

`PerceptionPerformanceAnalyzer.analyze()`によって各領域に対する解析結果が算出される．N 番目に追加した scene についてのみの解析結果を求めるには，`analyzer.analyze(scene=N)`とすることで解析結果が得られる．

```python
>>> score_df, error_df = analyzer.analyze()
>>> print(score_df)
                    TP   FP        FN  AP(Center Distance 3d [m])  APH(Center Distance 3d [m])  AP(IoU BEV)  APH(IoU BEV)  AP(IoU 3D)  APH(IoU 3D)  AP(Plane Distance [m])  APH(Plane Distance [m])
ALL         0.992003  0.0  0.007997                    0.972918                     0.970923     0.972918      0.970923    0.972918     0.970923                0.972918                 0.970923
car         1.000000  0.0  0.000000                    1.000000                     0.999460     1.000000      0.999460    1.000000     0.999460                1.000000                 0.999460
bicycle     0.958561  0.0  0.041439                    1.000000                     1.000000     1.000000      1.000000    1.000000     1.000000                1.000000                 1.000000
pedestrian  0.991278  0.0  0.008722                    0.900682                     0.896454     0.900682      0.896454    0.900682     0.896454                0.900682                 0.896454

>>> print(error_df)
ALL        x         1.394186  1.588129  7.605266e-01  2.300000  0.285734
           y         0.459921  0.611391  4.028307e-01  1.017925  0.000000
           yaw       0.188019  0.218617  1.115459e-01  0.390857  0.006516
           vx        0.016197  0.058724  5.644617e-02  0.635254  0.000000
           vy        0.062738  0.229145  2.203886e-01  1.721179  0.000000
           nn_plane  0.656631  0.902214  6.187298e-01  2.190708  0.000663
car        x         2.300000  2.300000  7.588600e-16  2.300000  2.300000
           y         0.000000  0.000000  0.000000e+00  0.000000  0.000000
           yaw       0.198764  0.198813  4.444152e-03  0.200147  0.182285
           vx        0.000000  0.000000  0.000000e+00  0.000000  0.000000
           vy        0.000000  0.000000  0.000000e+00  0.000000  0.000000
           nn_plane  0.779659  1.107139  7.860584e-01  2.116297  0.004645
bicycle    x         1.009088  1.154043  5.599603e-01  1.900625  0.310774
           y         0.559406  0.649149  3.293320e-01  0.908261  0.003252
           yaw       0.107742  0.143227  9.437081e-02  0.268970  0.006516
           vx        0.000000  0.000000  0.000000e+00  0.000000  0.000000
           vy        0.000000  0.000000  0.000000e+00  0.000000  0.000000
           nn_plane  0.564198  0.760262  5.095874e-01  1.926336  0.000663
pedestrian x         1.135335  1.324417  6.819782e-01  2.300000  0.285734
           y         0.752706  0.817092  3.179199e-01  1.017925  0.000000
           yaw       0.312139  0.319201  6.677537e-02  0.390857  0.178998
           vx        0.059606  0.112652  9.559111e-02  0.635254  0.000000
           vy        0.230876  0.439575  3.740625e-01  1.721179  0.000000
           nn_plane  0.688891  0.893696  5.693175e-01  2.190708  0.020005
```

## PerceptionPerformanceAnalyzer

| Arguments           |             type             | Mandatory |                    Description                    |
| :------------------ | :--------------------------: | :-------: | :-----------------------------------------------: |
| `evaluation_config` | `PerceptionEvaluationConfig` |    Yes    |                      config                       |
| `num_area_division` |            `int`             |    No     | 領域分割数(Options=[`1`, `3`, `9`]; Defaults=`1`) |

- `num_area_division`を指定することによって，自車周辺の領域を以下のように分割して，解析を行う．
  - `evaluation_config`内で`max_x_position`および`max_y_position`が指定されていない場合は，`100.0[m]`が適用される．

```python
        1:                            3:                  9:
                    max_x_position
                    +--------+          +--------+          +--------+
                    |    0   |          |____0___|          |_0|_1|_2|
    max_y_position  |    +   |          |____1___|          |_3|_4|_5|
                    |   ego  |          |    2   |          | 6| 7| 8|
                    +--------+          +--------+          +--------+
```

### Attributes

| name                |             type             | Description                    |
| :------------------ | :--------------------------: | :----------------------------- |
| `config`            | `PerceptionEvaluationConfig` | config                         |
| `df`                |      `pandas.DataFrame`      | 全 DataFrame                   |
| `plot_directory`    |            `str`             | プロット結果の保存ディレクトリ |
| `num_frame`         |            `int`             | 総フレーム数                   |
| `num_scene`         |            `int`             | 総シーン数                     |
| `num_area_division` |            `int`             | 領域の分割数                   |
| `num_ground_truth`  |            `int`             | Ground Truth 数                |
| `num_estimation`    |            `int`             | Estimation 数                  |
| `num_tp`            |            `int`             | TP 数                          |
| `num_fp`            |            `int`             | FP 数                          |
| `num_fn`            |            `int`             | FN 数                          |

### Basic methods

| name                   |                                          input                                          |       return       | Description                                                                                   |
| :--------------------- | :-------------------------------------------------------------------------------------: | :----------------: | :-------------------------------------------------------------------------------------------- |
| `get`                  |                                   `*args`, `**kwargs`                                   | `pandas.DataFrame` | `args`で指定したカラムまたは，`kwargs`で指定した条件の DataFrame                              |
| `sortby`               | `Union[str, List[str]]`, `df<Optional[pandas.DataFrame]>=None`, `ascending<bool>=False` | `pandas.DataFrame` | `Union[str, List[str]]`で指定したカラムに対してソートした DataFrame．`ascending=True`で昇順． |
| `head`                 |                                          `int`                                          | `pandas.DataFrame` | 先頭から`int(Defaults=5)`で指定した行数分の DataFrame                                         |
| `tail`                 |                                          `int`                                          | `pandas.DataFrame` | 末尾から`int(Defaults=5)`で指定した行数分の DataFrame                                         |
| `shape`                |                            `Optional[Union[str, List[str]]]`                            |    `Tuple[int]`    | 先頭から`int(Defaults=5)`で指定した行数の DataFrame                                           |
| `keys`                 |                                                                                         |     `pd.Index`     | `self.df`のカラム名                                                                           |
| `get_ground_truth`     |                      `df=<Optional[pandas.DataFrame]>`, `**kwargs`                      | `pandas.DataFrame` | Ground Truth の DataFrame                                                                     |
| `get_estimation`       |                      `df=<Optional[pandas.DataFrame]>`, `**kwargs`                      | `pandas.DataFrame` | Estimation の DataFrame                                                                       |
| `get_num_ground_truth` |                      `df=<Optional[pandas.DataFrame]>`, `**kwargs`                      |       `int`        | Ground Truth 数                                                                               |
| `get_num_estimation`   |                      `df=<Optional[pandas.DataFrame]>`, `**kwargs`                      |       `int`        | Estimation 数                                                                                 |
| `get_num_tp`           |                      `df=<Optional[pandas.DataFrame]>`, `**kwargs`                      |       `int`        | TP 数                                                                                         |
| `get_num_fp`           |                      `df=<Optional[pandas.DataFrame]>`, `**kwargs`                      |       `int`        | FP 数                                                                                         |
| `get_num_fn`           |                      `df=<Optional[pandas.DataFrame]>`, `**kwargs`                      |       `int`        | FN 数                                                                                         |

- `get()`では`*args`を指定することで指定した列を，`**kwargs`を指定することで指定した等号条件を満たす DataFrame を返す

```python
>>> analyzer = PerceptionPerformanceAnalyzer(...)

# 例) ラベル名がtruckのxy, uuidの列を参照したい場合
>>> analyzer.get("x", "y", "uuid", label="truck")
                         x          y                              uuid
0  ground_truth  85.536254   2.151734  a0e19d9fc8e528fb471d0d29bdf32927
   estimation    83.445015   2.306474                              None
1  ground_truth  82.125830   2.415408  a0e19d9fc8e528fb471d0d29bdf32927
   estimation    77.737808   2.873984                              None
2  ground_truth  49.703061   4.212113  a0e19d9fc8e528fb471d0d29bdf32927
   estimation    46.329472   4.206718                              None
3  ground_truth  27.750345   4.546522  a0e19d9fc8e528fb471d0d29bdf32927
   estimation    89.282616  -6.042823                              None
4  ground_truth  27.750345   4.546522  a0e19d9fc8e528fb471d0d29bdf32927
...
```

- `get_**()`では`**kwargs`で`COLUMN_NAME=value`指定することで特定のデータに対する結果を返す

```python
# 例) ラベル名がcarのground truth数を参照したい場合
>>> analyzer.get_num_ground_truth(label="car")
5223
```

### DataFrame 構造

- `add()`によって，各`PerceptionFrameResult`は以下のような形式で累積される．`scene`は`add()`もしくは`add_from_pkl()`をした際に追加された順番で 1~N が割り当てられる．

| index | type             | "timestamp" |   "x"   |   "y"   |   "w"   |   "l"   |   "h"   |  "yaw"  |  "vx"   |  "vy"   |  "nn_point1"   |  "nn_point2"   | "label" | "uuid" | "status" | "area" | "frame" | "scene" |
| ----: | :--------------- | :---------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :------------: | :------------: | :-----: | :----: | :------: | :----: | :-----: | :-----: |
|     0 | **ground_truth** |   `float`   | `float` | `float` | `float` | `float` | `float` | `float` | `float` | `float` | `tuple[float]` | `tuple[float]` |  `str`  | `str`  |  `str`   | `int`  |  `int`  |  `int`  |
|       | **estimation**   |             |         |         |         |         |         |         |         |         |                |                |         |        |          |        |         |         |

- `PerceptionPerformanceAnalyzer.df`で参照できる．

```python
>>> analyzer.df
                    timestamp          x          y         w          l         h       yaw            vx        vy  label                              uuid status  area  frame  scene
0  ground_truth  1.603763e+15  85.536254   2.151734  3.237000  12.112000  3.816000  0.017880  1.302359e-02  0.044080  truck  a0e19d9fc8e528fb471d0d29bdf32927     TP   0.0   48.0    1.0
   estimation    1.603763e+15  83.445015   2.306474  2.821630   6.807208  2.983142  0.030410  4.937477e-09  0.000000  truck                              None     TP   0.0   48.0    1.0
```

- Ground Truth のみ，Estimation のみも参照可能．

```python
>>> analyzer.get_ground_truth()
       timestamp          x         y      w       l      h       yaw        vx        vy                      nn_point1                       nn_point2 label                              uuid status  area  frame  scene
0   1.603763e+15  85.536254  2.151734  3.237  12.112  3.816  0.017880  0.013024  0.044080  (-7.172930, -49.5320, -0.2938), (-7.172930, -49.5320, -0.2938) truck  a0e19d9fc8e528fb471d0d29bdf32927     TP   0.0   48.0    1.0
1   1.603763e+15  82.125830  2.415408  3.237  12.112  3.816  0.020902 -0.001111  0.035555  (-7.172930, -49.5320, -0.2938), (-7.172930, -49.5320, -0.2938) truck  a0e19d9fc8e528fb471d0d29bdf32927     TP   0.0   49.0    1.0
...
```

```python
>>> analyzer.get_estimation()
      timestamp          x          y         w         l         h       yaw            vx   vy                      nn_point1                       nn_point2 label  uuid status  area  frame  scene
0  1.603763e+15  83.445015   2.306474  2.821630  6.807208  2.983142  0.030410  4.937477e-09  0.0  (-7.172930, -49.5320, -0.2938), (-7.172930, -49.5320, -0.2938) truck  None     TP   0.0   48.0    1.0
1  1.603763e+15  77.737808   2.873984  2.822067  6.618567  3.053950 -0.001258  4.937477e-09  0.0  (-7.172930, -49.5320, -0.2938), (-7.172930, -49.5320, -0.2938) truck  None     TP   0.0   49.0    1.0
...
```

### 解析

- `summarize_ratio()`
  - 各クラスに対する TP 率，FN 率，FP 率の算出

| label |  "TP"   |  "FN"   |  "FP"   |
| :---- | :-----: | :-----: | :-----: |
| `str` | `float` | `float` | `float` |

```python
>>> analyzer.summarize_ratio()
                  TP   FP        FN
All         0.992003  0.0  0.007997
car         1.000000  0.0  0.000000
pedestrian  0.991278  0.0  0.008722
motorbike   1.000000  0.0  0.000000
bicycle     0.958561  0.0  0.041439
```

- `summarize_score()`

| label |  "AP"   |  "APH"  | "MOTA"  | "MOTP"  | "IDswitch" |
| :---- | :-----: | :-----: | :-----: | :-----: | :--------: |
| `str` | `float` | `float` | `float` | `float` |   `int`    |

```python
>>> analyzer.summarize_score()
            AP(Center Distance 3d [m])  APH(Center Distance 3d [m])  AP(IoU BEV)  APH(IoU BEV)  AP(IoU 3D)  ...  MOTP(IoU 3D)  IDswitch(IoU 3D)  MOTA(Plane Distance [m])  MOTP(Plane Distance [m])  IDswitch(Plane Distance [m])
ALL                           0.972918                     0.963204     0.972918      0.963204    0.972918  ...           1.0                 0                  0.984711                       0.0                             0
car                           1.000000                     1.000000     1.000000      1.000000    1.000000  ...           1.0                 0                  1.000000                       0.0                             0
pedestrian                    0.900682                     0.900682     0.900682      0.900682    0.900682  ...           1.0                 0                  0.900682                       0.0                             0
motorbike                     0.990989                     0.952135     0.990989      0.952135    0.990989  ...           1.0                 0                  0.990989                       0.0                             0
bicycle                       1.000000                     1.000000     1.000000      1.000000    1.000000  ...           1.0                 0                  1.000000                       0.0                             0

```

- `summarize_error()`
  - 各クラスに対する average，RMS，std，max，min の算出

| label | element | "average" |  "rms"  |  "std"  |  "max"  |  "min"  |
| :---- | :------ | :-------: | :-----: | :-----: | :-----: | :-----: |
| `str` | `str`   |  `float`  | `float` | `float` | `float` | `float` |

```python
>>> analyzer.summarize_error()
ALL        x         1.394186  1.588129  7.605266e-01  2.300000  0.285734
           y         0.459921  0.611391  4.028307e-01  1.017925  0.000000
           yaw       0.188019  0.218617  1.115459e-01  0.390857  0.006516
           vx        0.016197  0.058724  5.644617e-02  0.635254  0.000000
           vy        0.062738  0.229145  2.203886e-01  1.721179  0.000000
           nn_plane  0.656631  0.902214  6.187298e-01  2.190708  0.000663
car        x         2.300000  2.300000  7.588600e-16  2.300000  2.300000
           y         0.000000  0.000000  0.000000e+00  0.000000  0.000000
           yaw       0.198764  0.198813  4.444152e-03  0.200147  0.182285
           vx        0.000000  0.000000  0.000000e+00  0.000000  0.000000
           vy        0.000000  0.000000  0.000000e+00  0.000000  0.000000
           nn_plane  0.779659  1.107139  7.860584e-01  2.116297  0.004645
bicycle    x         1.009088  1.154043  5.599603e-01  1.900625  0.310774
           y         0.559406  0.649149  3.293320e-01  0.908261  0.003252
           yaw       0.107742  0.143227  9.437081e-02  0.268970  0.006516
           vx        0.000000  0.000000  0.000000e+00  0.000000  0.000000
           vy        0.000000  0.000000  0.000000e+00  0.000000  0.000000
           nn_plane  0.564198  0.760262  5.095874e-01  1.926336  0.000663
pedestrian x         1.135335  1.324417  6.819782e-01  2.300000  0.285734
           y         0.752706  0.817092  3.179199e-01  1.017925  0.000000
           yaw       0.312139  0.319201  6.677537e-02  0.390857  0.178998
           vx        0.059606  0.112652  9.559111e-02  0.635254  0.000000
           vy        0.230876  0.439575  3.740625e-01  1.721179  0.000000
           nn_plane  0.688891  0.893696  5.693175e-01  2.190708  0.020005
```

### プロット関数

- `plot_by_time()`

  - 指定した GT オブジェクトに対し位置または速度と誤差を時系列で描画

    | Arguments |  type  | Mandatory |                       Description                        |
    | :-------- | :----: | :-------: | :------------------------------------------------------: |
    | `uuid`    | `str`  |    Yes    |              対象 GT オブジェクトの uuid．               |
    | `column`  | `str`  |    Yes    |    位置または速度を指定．(Options=[`xy`, `velocity`])    |
    | `scene`   | `int`  |    No     | 対象シーン．未指定の場合，最後に追加されたシーンが使用． |
    | `show`    | `bool` |    No     |    描画結果を表示するかのフラッグ(Defaults=`False`)．    |

    ```python
    # 例: uuid: "4bae7e75c7de70be980ce20ce8cbb642"のオブジェクトのxyについてプロット

    >> analyzer.plot_by_time("4bae7e75c7de70be980ce20ce8cbb642", ["x", "y"])
    ```

    <img src="./figure/sample_plot_by_time.png" width=800 height=400>

- `plot_num_objects()`

  - `base_link`からの距離ごとのオブジェクト数をヒストグラムでプロット

  | Arguments  |             type             | Mandatory |                           Description                            |
  | :--------- | :--------------------------: | :-------: | :--------------------------------------------------------------: |
  | `area_idx` |            `int`             |    No     |  対象分割領域の番号．未指定の場合，全領域のオブジェクトを描画．  |
  | `label`    | `Union[str, AutowareLabel]`  |    No     |    対象ラベル名．未指定の場合，全ラベルのオブジェクトを描画．    |
  | `status`   | `Union[str, MatchingStatus]` |    No     | 対象マッチングステータス名．未指定の場合，全オブジェクトを描画． |
  | `dist_bin` |           `float`            |    No     |                   距離の間隔(Defaults=`0.5`)．                   |
  | `show`     |            `bool`            |    No     |        描画結果を表示するかのフラッグ(Defaults=`False`)．        |

  ```python
  # 全オブジェクトの数をプロット
  >> analyzer.plot_num_objects()
  ```

  <img src="./figure/sample_plot_num_objects.png" width=800 height=400>
