
## FAQ

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
