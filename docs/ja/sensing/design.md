## データ構造

### 点群の検知結果

1. ~Bounding box の中の点群の座標値 (List[Tuple[float]])~
   - pros: 判断根拠を残せる
   - cons: 計算量多、対して使わないのにデータ量が多くて邪魔
2. **Bounding box の中の点群の個数 (int)**
   - pros: ある程度の判断根拠を残せる、result.json は見やすい
   - cons: 計算量多
3. ~Bounding box の中の点群の有無 (bool)~
   - pros: result.json は見やすい、計算量少
   - cons: 点群の様子を見たいときに可視化が必要になる
   - perception_eval では 2 を採用、理由としては
     - 1.に関しては、細かい点群の位置は、viewer や rosbag で確認できれば良い
     - 現状のの Sensing lsim v1 で 2 が採用されている

### 評価単位でのデータ構造

- SensingEvaluationConfig: 評価全体に関する config, シナリオファイルを読み込むことで設定する[(参照)](../../../perception_eval/perception_eval/config/sensing_evaluation_config.py)

  ```txt
    Arguments:
      dataset_paths (List[str]): データセットのパス
      does_use_pointcloud (bool): データセットの点群を読み込むかどうか
      result_root_directory (str): 評価結果の保存ディレクトリパス
      log_directory (str): ログの保存ディレクトリパス
      visualization_directory (str): 可視化結果の保存ディレクトリパス
      target_uuids (List[str]): 検出対象objectのuuidのリスト
      box_scale_0m (float): 0m位置でのbounding boxのスケーリング係数
      box_scale_100m (float): 100m位置でのbounding boxのスケーリング係数
      min_points_threshold (int): bounding box内で検出されるべき最低限の点群数の閾値
  ```

- SensingEvaluationManager: 評価全体のための manager[(参照)](../../../perception_eval/perception_eval/manager/perception_evaluation_manager.py)

### Frame 単位でのデータ構造

- SensingFrameResult: 1 frame に対しての検出・非検出対象エリアに対する点群のセンシング結果[(参照)](../../../perception_eval/perception_eval/evaluation/sensing/sensing_frame_result.py)

  ```txt
    Arguments:
        sensing_frame_config (SensingFrameConfig): Frame単位での評価用のconfig
        unix_time (int): Unix time
        frame_name (str): Frame名

    Attributes:
        sensing_frame_config (SensingFrameConfig): 同上
        unix_time (int): 同上
        frame_name (str): 同上
        *detection_success_results (List[DynamicObjectWithSensingResult]): 各検出対象エリアで検出成功したobjectのリスト
        *detection_fail_results (List[DynamicObjectWithSensingResult]): 各検出対象エリアで検出失敗したobjectのリスト
        *pointcloud_failed_non_detected: (List[numpy.ndarray]): 各非検出対象エリアごとで非検出失敗（=誤検出）された点群（リストの長さ = 非検出対象エリア数）
  ```

  (\*)は，SensingFrameResult のインスタンス生成時には空のリスト，

  ```python
  evaluate_frame(
    ground_truth_objects: List[DynamicObject],
    pointcloud_for_detection: numpy.ndarray,
    pointcloud_for_non_detection: List[numpy.ndarray]
  )
  ```

  メソッドを呼ぶことで評価が実行される．

- SensingFrameConfig: Frame 単位での評価用の config[(参照)](../../../perception_eval/perception_eval/evaluation/sensing/sensing_frame_config.py)

  ```txt
    Arguments:
      bbox_scale_0m (float): 0m地点でのBounding boxのスケーリング係数
      bbox_scale_100m (float): 100m地点でのBounding boxのスケーリング係数
      min_points_threshold (int): bounding box内で検出されるべき最低限の点群数の閾値

    Attributes:
      bbox_scale_0m (float): 0m地点でのBounding boxのスケーリング係数
      bbox_scale_100m (float): 100m地点でのBounding boxのスケーリング係数
  ```

  - 各位置にある bounding box のスケーリング係数は`get_scale_factor(distance: float)`メソッドによって線形に計算され取得できる

### Object 単位でのデータ構造

- DynamicObjectWithSensingResult: 1 ground truth object（アノテーションされた bounding box）に対しての結果[(参照)](../../../perception_eval/perception_eval/evaluation/sensing/sensing_result.py)

  ```txt
    Arguments:
      ground_truth_object (DynamicObject): Ground truth object
      pointcloud (numpy.ndarray): 地面除去された点群
      scale_factor (float): Bounding box のスケーリング係数

    Attributes:
      ground_truth_object (DynamicObject): 同上
      inside_pointcloud: (numpy.ndarray): Bounding box内の点群
      inside_pointcloud_num (int): Bounding box内の点群数
      is_detected (bool): Bounding box内に１つでも点があればTrue, 0ならばFalse
  ```
