# [`<class> PerceptionAnalyzer3D(...)`](../../../perception_eval/perception_eval/tool/perception_analyzer3d.py)

3D Perception の評価結果をもとに解析を行う．

## Perception 評価種別

評価・解析方法として，UseCase/DataBase の 2 種類がある．

- UseCase 評価 = 設計されたテストケースにおいて，対象物体に着目してメトリクスの評価・PASS/FAIL の判定を行う

  - 認識性能の精度部分を評価，安全性能の担保をメインの目的とした評価

- DataBase 評価 = 実環境で収集したデータに対して，視野内の物体に対してメトリクスの評価を行う

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
from perception_eval.tool import PerceptionAnalyzer3D


# 1. with PerceptionEvaluationConfig
# REQUIRED:
#   - evaluation_config <PerceptionEvaluationConfig>

analyzer = PerceptionAnalyzer3D(evaluation_config)

# 2. with scenario file (.yaml)
# REQUIRED:
#   - scenario_path <str>

analyzer = PerceptionAnalyzer3D.from_scenario(result_root_directory, scenario_path)
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

`PerceptionAnalyzer3D.analyze()`によって各領域に対する解析結果が算出される．N 番目に追加した scene についてのみの解析結果を求めるには，`analyzer.analyze(scene=N)`とすることで解析結果が得られる．

```python
>>> result = analyzer.analyze()
>>> print(result.score)
                    TP   FP        FN  AP(Center Distance 3d [m])  APH(Center Distance 3d [m])  AP(IoU BEV)  APH(IoU BEV)  AP(IoU 3D)  APH(IoU 3D)  AP(Plane Distance [m])  APH(Plane Distance [m])
ALL         0.992003  0.0  0.007997                    0.972918                     0.970923     0.972918      0.970923    0.972918     0.970923                0.972918                 0.970923
car         1.000000  0.0  0.000000                    1.000000                     0.999460     1.000000      0.999460    1.000000     0.999460                1.000000                 0.999460
bicycle     0.958561  0.0  0.041439                    1.000000                     1.000000     1.000000      1.000000    1.000000     1.000000                1.000000                 1.000000
pedestrian  0.991278  0.0  0.008722                    0.900682                     0.896454     0.900682      0.896454    0.900682     0.896454                0.900682                 0.896454

>>> print(result.error)
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

## `<class> PerceptionAnalyzer3D(...)`

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

| name                     |                                          input                                          |       return       | Description                                                                                   |
| :----------------------- | :-------------------------------------------------------------------------------------: | :----------------: | :-------------------------------------------------------------------------------------------- |
| `get()`                  |                                   `*args`, `**kwargs`                                   | `pandas.DataFrame` | `args`で指定したカラムまたは，`kwargs`で指定した条件の DataFrame                              |
| `sortby()`               | `Union[str, List[str]]`, `df<Optional[pandas.DataFrame]>=None`, `ascending<bool>=False` | `pandas.DataFrame` | `Union[str, List[str]]`で指定したカラムに対してソートした DataFrame．`ascending=True`で昇順． |
| `head()`                 |                                          `int`                                          | `pandas.DataFrame` | 先頭から`int(Defaults=5)`で指定した行数分の DataFrame                                         |
| `tail()`                 |                                          `int`                                          | `pandas.DataFrame` | 末尾から`int(Defaults=5)`で指定した行数分の DataFrame                                         |
| `shape()`                |                            `Optional[Union[str, List[str]]]`                            |    `Tuple[int]`    | 先頭から`int(Defaults=5)`で指定した行数の DataFrame                                           |
| `keys()`                 |                                                                                         |     `pd.Index`     | `self.df`のカラム名                                                                           |
| `get_ground_truth()`     |                      `df=<Optional[pandas.DataFrame]>`, `**kwargs`                      | `pandas.DataFrame` | Ground Truth の DataFrame                                                                     |
| `get_estimation()`       |                      `df=<Optional[pandas.DataFrame]>`, `**kwargs`                      | `pandas.DataFrame` | Estimation の DataFrame                                                                       |
| `get_num_ground_truth()` |                      `df=<Optional[pandas.DataFrame]>`, `**kwargs`                      |       `int`        | Ground Truth 数                                                                               |
| `get_num_estimation()`   |                      `df=<Optional[pandas.DataFrame]>`, `**kwargs`                      |       `int`        | Estimation 数                                                                                 |
| `get_num_tp()`           |                      `df=<Optional[pandas.DataFrame]>`, `**kwargs`                      |       `int`        | TP 数                                                                                         |
| `get_num_fp()`           |                      `df=<Optional[pandas.DataFrame]>`, `**kwargs`                      |       `int`        | FP 数                                                                                         |
| `get_num_fn()`           |                      `df=<Optional[pandas.DataFrame]>`, `**kwargs`                      |       `int`        | FN 数                                                                                         |
| `get_ego2map()`          |                              `scene=<int>`, `frame=<int>`                               |  `numpy.ndarray`   | 対象データを base_link->map 座標系に変換する 4x4 同次変換行列                                 |

- `get()`では`*args`を指定することで指定した列を，`**kwargs`を指定することで指定した等号条件を満たす DataFrame を返す

```python
>>> analyzer = PerceptionAnalyzer3D(...)

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
  - なお，x,y,yaw,vx,vy は base_link 座標系に従っている．

| index | type             | "timestamp" |   "x"   |   "y"   |   "w"   |   "l"   |   "h"   |  "yaw"  |  "vx"   |  "vy"   |  "nn_point1"   |  "nn_point2"   | "label" | "confidence" | "uuid" | "num_points" | "status" | "area" | "frame" | "scene" |
| ----: | :--------------- | :---------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :------------: | :------------: | :-----: | :----------: | :----: | :----------: | :------: | :----: | :-----: | :-----: |
|     0 | **ground_truth** |   `float`   | `float` | `float` | `float` | `float` | `float` | `float` | `float` | `float` | `tuple[float]` | `tuple[float]` |  `str`  |   `float`    | `str`  |    `int`     |  `str`   | `int`  |  `int`  |  `int`  |
|       | **estimation**   |

- `PerceptionAnalyzer3D.df`で参照できる．

```python
>>> analyzer.df
                      timestamp          x         y         w          l  ...
0    ground_truth  1.603763e+15  45.108863  4.415448  3.237000  12.112000
     estimation    1.603763e+15  42.187082  4.216309  2.818074   6.961135
1    ground_truth  1.603763e+15  41.231904  4.528642  3.237000  12.112000
     estimation    1.603763e+15  37.909447  4.304737  2.694330   6.547065
...
```

- Ground Truth のみ，Estimation のみも参照可能．

```python
>>> analyzer.get_ground_truth()
         timestamp          x         y         w       l         h       yaw ...
0     1.603763e+15  45.108863  4.415448  3.237000  12.112  3.816000  0.056254
1     1.603763e+15  41.231904  4.528642  3.237000  12.112  3.816000  0.059814
2     1.603763e+15  38.064378  4.594842  3.237000  12.112  3.816000  0.062258
...
```

```python
>>> analyzer.get_estimation()
         timestamp          x         y         w       l         h       yaw ...
0     1.603763e+15  45.108863  4.415448  3.237000  12.112  3.816000  0.056254
1     1.603763e+15  41.231904  4.528642  3.237000  12.112  3.816000  0.059814
2     1.603763e+15  38.064378  4.594842  3.237000  12.112  3.816000  0.062258
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

### プロットメソッド関数

- `<enum> PlotAxes`

  - プロットの変数軸を指定するためのクラス

    | Member       | Description                |
    | :----------- | :------------------------- |
    | `FRAME`      | フレーム番号               |
    | `TIME`       | 時間[s]                    |
    | `DISTANCE`   | 距離[m]                    |
    | `X`          | x 座標[m]                  |
    | `Y`          | y 座標[m]                  |
    | `VX`         | x 方向速度[m/s]            |
    | `VY`         | y 方向速度[m/s]            |
    | `CONFIDENCE` | 推定値の信頼度[0, 1]       |
    | `POSITION`   | (x,y)座標[m]               |
    | `VELOCITY`   | (vx, vy)速度[m/s]          |
    | `POLAR`      | 極座標系(theta[rad], r[m]) |

- `<func> plot_state(...) -> None`

  - 指定した GT オブジェクトに対し位置，向き，速度の状態を描画

    | Arguments |                  type                  | Mandatory | Description                                                           |
    | :-------- | :------------------------------------: | :-------: | :-------------------------------------------------------------------- |
    | `uuid`    |                 `str`                  |    Yes    | 対象 GT オブジェクトの uuid．                                         |
    | `columns` |        `Union[str, List[str]]`         |    Yes    | 位置または速度を指定(Options=[`x`, `y`, `yaw`, `w`, `l`, `vx`, `vy`]) |
    | `mode`    |               `PlotAxes`               |    No     | プロット変数の軸(Defaults=`PlotAxes.TIME`)                            |
    | `status`  | `Optional[Union[str, MatchingStatus]]` |    No     | 対象オブジェクトのマッチング状態．(Defaults=`None`)                   |
    | `show`    |                 `bool`                 |    No     | 描画結果を表示するかのフラッグ(Defaults=`False`)                      |

    ```python
    # 例: uuid: "4bae7e75c7de70be980ce20ce8cbb642"のオブジェクトのxyについて時系列でTP/FP/FNに関わらずプロット
    >> analyzer.plot_state("4bae7e75c7de70be980ce20ce8cbb642", ["x", "y"])
    ```

    <img src="../../fig/perception/plot_state_by_time.png" width=800>

- `<func> plot_error(...) -> None`

  - 指定した状態量に対する GT と推定値の誤差をプロット

    | Arguments |          type           | Mandatory | Description                                                           |
    | :-------- | :---------------------: | :-------: | :-------------------------------------------------------------------- |
    | `columns` | `Union[str, List[str]]` |    Yes    | 位置または速度を指定(Options=[`x`, `y`, `yaw`, `w`, `l`, `vx`, `vy`]) |
    | `mode`    |       `PlotAxes`        |    No     | 時系列または距離ごと(Defaults=`PlotAxes.TIME`)                        |
    | `show`    |         `bool`          |    No     | 描画結果を表示するかのフラッグ(Defaults=`False`)                      |

    ```python
    # 例: xyの誤差について時系列でプロット
    >> analyzer.plot_error(["x", "y"])
    ```

    <img src="../../fig/perception/plot_error_by_time.png" width=800>

- `<func> plot_num_object(...) -> None`

  - GT/Estimation のオブジェクト数をヒストグラムでプロット

  | Arguments  |      type       | Mandatory | Description                                                                       |
  | :--------- | :-------------: | :-------: | :-------------------------------------------------------------------------------- |
  | `mode`     |   `PlotAxes`    |    No     | 距離ごとまたは時系列(Defaults=`PlotAxes.DISTANCE`)                                |
  | `bins`     | `Optional[int]` |    No     | 描画間隔．デフォルトで距離ごとなら`10`[m]，時系列なら`100`[ms]．(Defaults=`None`) |
  | `heatmap`  |     `bool`      |    No     | ヒートマップを可視化するかどうか(3Dのみ)(Defaults=`False`)                        |
  | `show`     |     `bool`      |    No     | 描画結果を表示するかのフラッグ(Defaults=`False`)                                  |
  | `**kwargs` |      `Any`      |    No     | 特定のラベル，エリアなどについてのみ描画する際に指定．(e.g.`area=0`)              |

  ```python
  # 全オブジェクトの数を距離ごとにプロット
  >> analyzer.plot_num_object()
  ```

  <img src="../../fig/perception/plot_num_object_by_distance.png" width=800 height=400>

- `<func> box_plot(...) -> None`

  - 指定した状態に対する誤差の箱ひげ図をプロット

  | Arguments |          type          | Mandatory | Description                                      |
  | :-------- | :--------------------: | :-------: | :----------------------------------------------- |
  | `columns` | `Union[str, List[str]` |    Yes    | 描画対象となる誤差(x, y, yaw, w, l, vx, vy)      |
  | `show`    |         `bool`         |    No     | 描画結果を表示するかのフラッグ(Defaults=`False`) |

  ```python
  # x, yの誤差について箱ひげ図をプロット
  >> analyzer.box_plot(["x", "y"])
  ```

  <img src="../../fig/perception/box_plot_xy.png" width=400>

## Known issues / Limitations

- `PerceptionAnalyzer3D()`は 3D 評価のみ対応
  <img src="../../fig/perception/plot_num_object_by_distance.png" width=800>

## [`<class> Gmm(...)`](../../../perception_eval/perception_eval/tool/gmm.py)

[`sklearn.mixture.GaussianMixture`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)のラッパークラス．
同時確率分布 P(X, Y)のパラメータを EM アルゴリズムを用いて推定し，入力データ X から事後確率分布 P(Y|X)の平均値を予測する．
このとき，X は状態量，Y は誤差の状態量を表す．

- Initialization

  | Arguments      | type  | Description                      |
  | :------------- | :---: | :------------------------------- |
  | `max_k`        | `int` | クラスタ数 K の最大値.           |
  | `n_init`       | `int` | クラスタ数の初期値. (Default: 1) |
  | `random_state` | `int` | 乱数シードの値. (Default=1234)   |

- Methods

  | Methods           |     Returns     | Description                                      |
  | :---------------- | :-------------: | :----------------------------------------------- |
  | `fit()`           |     `None`      | EM アルゴリズムを用いてモデルのパラメータを推定. |
  | `predict()`       | `numpy.ndarray` | 入力データ X の事後確立分布の平均値を予測.       |
  | `predict_label()` | `numpy.ndarray` | 入力データ X の各ラベルを予測.                   |
  | `save()`          |     `None`      | 推定された最良モデルを pickle で保存.            |
  | `load()`          |      `GMM`      | pickle で保存されたモデルをロード.               |
  | `plot_ic()`       |     `None`      | AIC と BIC の値をプロット.                       |

### `<func> load_sample(...) -> Tuple[numpy.ndarray, numpy.ndarray]`

- Returns input array of specified states and errors.

| Arguments  |          type          | Mandatory | Description                            |
| :--------- | :--------------------: | :-------: | :------------------------------------- |
| `analyzer` | `PerceptionAnalyzer3D` |    Yes    | `PerceptionAnalyzer3D` のインスタンス. |
| `state`    |      `List[str]`       |    Yes    | 対象となる状態量名のリスト.            |
| `error`    |      `List[str]`       |    Yes    | 対象となる誤差名のリスト.              |

### Example usage

```python
from perception_eval.tool import PerceptionAnalyzer3D, Gmm, load_sample
import numpy as np

# PerceptionAnalyzer3Dの初期化
analyzer = PerceptionAnalyzer3D(...)

# サンプルのロード, X: state, Y: error
state = ["x", "y", "yaw", "vx", "xy"]
error = ["x", "y"]

# X: (N, 5), Y: (N, 2)
X, Y = load_sample(analyzer, state, error)
sample = np.concatenate([X, Y], axis=-1)  # (N, 7)

# モデルパラメータの推定
model = Gmm(max_k)
model.fit(sample)

# X(状態量)からY(誤差)を予測．
y_pred = model.predict(X)  # (N, 2)
```
