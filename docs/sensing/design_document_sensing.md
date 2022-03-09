## データ構造
### 点群の検知結果

1. Bounding boxの中の点群の座標値 (List[Tuple[float]])
  - pros: 判断根拠を残せる
  - cons: 計算量多、対して使わないのにデータ量が多くて邪魔
2. Bounding boxの中の点群の個数 (int)
  - pros: ある程度の判断根拠を残せる、result.jsonは見やすい
  - cons: 計算量多
3. Bounding boxの中の点群の有無 (bool)
  - pros: result.jsonは見やすい、計算量少
  - cons: 点群の様子を見たいときに可視化が必要になる
- awml_evaluationでは2を採用
  - 1.に関しては、細かい点群の位置は、viewerやrosbagで確認できれば良い
  - 現状ののSensing lsim v1で2が採用されている
  - が理由
