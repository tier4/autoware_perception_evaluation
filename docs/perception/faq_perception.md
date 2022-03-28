## FAQ for error

### rosbag を再生する度に検出結果が大きく変わる

- 以下のように rosbag を再生する度に検出結果が変わる場合
  - 以下の例だと Ap が 0.05 程度ずれている
  - 2021/12/08 現在、Ap は 0.002 ぐらいしか振れない

```yaml
1回目
[perception_evaluator_node.py-56] mAP: 0.099, mAPH: 0.087 (center distance 3d [m])
[perception_evaluator_node.py-56] AP , car ([1.0]): 0.394, bicycle ([1.0]): 0.000, pedestrian ([1.0]): 0.000, motorbike ([1.0]): 0.000

2回目
[perception_evaluator_node.py-56] mAP: 0.114, mAPH: 0.102 (center distance 3d [m])
[perception_evaluator_node.py-56] AP , car ([1.0]): 0.456, bicycle ([1.0]): 0.000, pedestrian ([1.0]): 0.000, motorbike ([1.0]): 0.000
```

- 解決法
  - <https://tier4.atlassian.net/wiki/spaces/AIP/pages/1436025612/cyclonedds> これを参考に CycloneDDS の設定を行う
- 解決できているかチェック
  - ノート PC だと rate 0.1 ぐらいにすると安牌
  - 1frame の間に width が数万レベルで変化していたら concat の点群が抜け落ちている
  - その場合通信の設定が正しいか見直す

```yaml
ros2 bag play ./input_bag/ --clock 100 --rate 0.25
ros2 topic echo /sensing/lidar/concatenated/pointcloud --no-arr
```

```yaml
header:
  stamp:
    sec: 1624164492
    nanosec: 350359040
  frame_id: base_link
height: 1
width: 215055
fields: "<sequence type: sensor_msgs/msg/PointField, length: 4>"
is_bigendian: false
point_step: 32
row_step: 1079360
data: "<sequence type: uint8, length: 6881760>"
is_dense: true
```

- よくあるミスその 1：再起動時に必要なコマンドを忘れていないか

```bash
sudo sysctl -w net.core.rmem_max=2147483647
sudo ifconfig lo multicast
```

- 2021/12/08 現在
  - rate を落として rosbag play をすると concat pointcloud までは再現するようになる
  - centerpoint の出力で 20%の frame で 30%くらいの object の個数がずれる（計 6%程度はずれる）までは抑えられることを確認済

## FAQ for usage

### Q. predicted_object とは何か

- predicted_object = Autoware の推論結果のこと

### Q. Plane distance とは何か

- メトリクスにおける TP/FP の判定において，Usecase 評価で Ground truth object と Predicted object の**自車近傍の 2 点の距離の RMS**を以って判定する．具体的には，

1. GT と Det それぞれにおいて，footprint の端点のうち Ego から近い面(=2 点)を選択する．
2. その面同士における 2 通りの端点のペアから，合計距離が短いペアを選択し，これを自車近傍の 2 点する．
3. 各ペアの距離の 2 乗平均平方根をとり，これを\*自車近傍の 2 点の距離の RMS\*\*と呼ぶ．

- 例

1. GT において，Ego から近い面として面 g3g4 を選択する．Det においては，面 d3d4 を選択する．
2. 端点のペアは，(g3d3, g4d4)と(g3d4, g4d3)の 2 通りある．合計距離が短いペアを選択する．図例では，(g3d3, g4d4)を選択する．
3. 自車近傍の 2 点の距離の RMS = sqrt ( ( g3d3^2 + g4d4^2 ) / 2 )
   - 詳しくは，`get_uc_plane_distance`関数を参照
   - 1 の背景：検出された物体の奥行きが不明瞭なので，確度の高い自車近傍の点を選択している．
     ![pipeline](figure/uc_plane_distance.svg)

- なぜか各 rosbag ごとに（crop_box_filter を変更させて record して）点群の最大距離が異なる -> 検出能力が変わっているので PerceptionEvaluationConfig を変えて評価

### Q. uuid で具体的にできることはなにか？

- uuid = object を一意に定めるための id
  - Autoware だと uuid <https://github.com/tier4/autoware_iv_msgs/blob/main/autoware_perception_msgs/msg/object_recognition/DynamicObject.msg#L1>の表現を行う
  - nuscenes の表現だと instance_token という名称
- 全 Frame の大量の ObjectResult (object ごとの結果）の結果に対して、「object "79151c9c4ebc7380555f25aecc031422" の結果は？」と投げると、その object の
  - ground truth との誤差（center disntace, 面距離等々）
  - pointcloud が何点当たっているか
  - などの時間推移が表示できるようになる

### Q. sensor_model, vehicle_model, vehicle_id が変わっても、PerceptionEvaluationConfig は共通で良いのか

- database 評価(= PerceptionEvaluationConfig で設定するもの）に関しては、基本的に検出能力が変わらない限り同じ config で良いと思っている
- 例えば
  - センサの数が変わりました -> 検出能力が変わっている（例えば検出距離が変わる）ので PerceptionEvaluationConfig を変えて評価（=database 評価したい n 個の rosbag のセンサ構成は同じであってほしい）
  - calibration し直しました -> 検出能力は買わないはずなので同じ設定で良い
  - 異なる車 -> センサが同じ配置なら検出能力は変わらないはず
  - （レアケース）各 rosbag ごとに（crop_box_filter を変更させて record して）点群の最大距離が異なる -> 検出能力が変わっているので EvaluationConfig を変えて評価
