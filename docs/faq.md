
## FAQ

### rosbagを再生する度に検出結果が大きく変わる

- 以下のようにrosbagを再生する度に検出結果が変わる場合

```
1回目 [perception_evaluator_node.py-55] mAP (center_distance [m]): 0.188 (num 1944)
2回目 [perception_evaluator_node.py-55] mAP (center_distance [m]): 0.230 (num 1866)
```

- <https://tier4.atlassian.net/wiki/spaces/AIP/pages/1436025612/cyclonedds> これを参考にCycloneDDSの設定を行う
- 2021/12/08現在
  - rateを落としてrosbag playをするとconcat pointcloudまでは再現するようになるはず
  - centerpointの出力で20%のframeで30%くらいのobjectの個数がずれる（計6%程度はずれる）までは抑えられることを確認した

