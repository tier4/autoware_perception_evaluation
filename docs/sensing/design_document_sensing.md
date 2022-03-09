## データ構造

### 点群の検知結果

1. Bounding box の中の点群の座標値 (List[Tuple[float]])
   - pros: 判断根拠を残せる
   - cons: 計算量多、対して使わないのにデータ量が多くて邪魔
2. Bounding box の中の点群の個数 (int)
   - pros: ある程度の判断根拠を残せる、result.json は見やすい
   - cons: 計算量多
3. Bounding box の中の点群の有無 (bool)
   - pros: result.json は見やすい、計算量少
   - cons: 点群の様子を見たいときに可視化が必要になる
   - awml_evaluation では 2 を採用、理由としては
     - 1.に関しては、細かい点群の位置は、viewer や rosbag で確認できれば良い
     - 現状のの Sensing lsim v1 で 2 が採用されている
