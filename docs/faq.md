
## FAQ for error

### rosbagを再生する度に検出結果が大きく変わる

- 以下のようにrosbagを再生する度に検出結果が変わる場合
  - 以下の例だとApが0.05程度ずれている
  - 2021/12/08現在、Apは0.002ぐらいしか振れない

```
1回目
[perception_evaluator_node.py-56] mAP: 0.099, mAPH: 0.087 (center distance 3d [m])
[perception_evaluator_node.py-56] AP , car ([1.0]): 0.394, bicycle ([1.0]): 0.000, pedestrian ([1.0]): 0.000, motorbike ([1.0]): 0.000

2回目
[perception_evaluator_node.py-56] mAP: 0.114, mAPH: 0.102 (center distance 3d [m])
[perception_evaluator_node.py-56] AP , car ([1.0]): 0.456, bicycle ([1.0]): 0.000, pedestrian ([1.0]): 0.000, motorbike ([1.0]): 0.000
```

- 解決法
  - <https://tier4.atlassian.net/wiki/spaces/AIP/pages/1436025612/cyclonedds> これを参考にCycloneDDSの設定を行う
- 解決できているかチェック
  - ノートPCだとrate 0.1ぐらいにすると安牌
  - 1frameの間にwidthが数万レベルで変化していたらconcatの点群が抜け落ちている
  - その場合通信の設定が正しいか見直す

```
ros2 bag play ./input_bag/ --clock 100 --rate 0.25
ros2 topic echo /sensing/lidar/concatenated/pointcloud --no-arr
```

```
header:
  stamp:
    sec: 1624164492
    nanosec: 350359040
  frame_id: base_link
height: 1
width: 215055
fields: '<sequence type: sensor_msgs/msg/PointField, length: 4>'
is_bigendian: false
point_step: 32
row_step: 1079360
data: '<sequence type: uint8, length: 6881760>'
is_dense: true

```

- よくあるミスその1：再起動時に必要なコマンドを忘れていないか

```
sudo sysctl -w net.core.rmem_max=2147483647
sudo ifconfig lo multicast
```

- 2021/12/08現在
  - rateを落としてrosbag playをするとconcat pointcloudまでは再現するようになる
  - centerpointの出力で20%のframeで30%くらいのobjectの個数がずれる（計6%程度はずれる）までは抑えられることを確認済

## FAQ for usage

### Q. sensor_model, vehicle_model, vehicle_idが変わっても、EvaluationConfigは共通で良いのか

- A. database評価(= EvaluationConfigで設定するもの）に関しては、基本的に検出能力が変わらない限り同じconfigで良いかなと思っています
- 例えば
  - センサの数が変わりました -> 検出能力が変わっている（例えば検出距離が変わる）のでEvaluationConfigを変えて評価（=database評価したいn個のrosbagのセンサ構成は同じであってほしい）
  - calibrationし直しました -> 検出能力は買わないはずなので同じ設定で良い
  - 異なる車 -> センサが同じ配置なら検出能力は変わらないはず
  - なぜか各rosbagごとに（crop_box_filterを変更させてrecordして）点群の最大距離が異なる -> 検出能力が変わっているのでEvaluationConfigを変えて評価
