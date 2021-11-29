## Case 1. rosbagを用いずにデータセットのみで評価を行う

- 評価Metricsの開発などに
- poetry install

```
pip3 install poetry
```

- setting

```
git clone https://github.com/tier4/AWMLevaluation.git
cd AWMLevaluation
poetory update
```

- 実行

```
cd awml_evaluation
poetry run python3 -m test.lsim
```


## Case 2. Lsim側に組み込む
### install

- pip, submodule等は使わない
- reposに追加
  - 評価地点に関してはhashで管理する

```
repositories:
  # autoware
  simulator/AWMLevaluation:
    type: git
    url: git@github.com/tier4/AWMLevaluation.git
    version: main
```

### Managerを起動

```python
        self.evaluator: EvaluationManager = EvaluationManager(
            dataset_path=dataset_path,
            does_use_pointcloud=False,
            result_root_directory="data/result/{TIME}/",
            log_directory="",
            visualization_directory="visualization/",
            evaluation_tasks=["detection"],
            target_labels=["car", "bicycle", "pedestrian", "motorbike"],
            map_thresholds_center_distance=[0.5, 1.0, 2.0],
            map_thresholds_plane_distance=[0.5, 1.0, 2.0],
            map_thresholds_iou=[],
        )
```

### UseCase評価について

```python

class Lsim:
    def __init__(self):
        self.evaluator = EvaluationManager()

    def callback():
        # ROS topic awmlevaluatorのobject形式
        predicted_objects : List[DynamicObject] = set_from_ros_topic(objects_from_topic)

        # 対応するGround truthを取得
        ground_truth_now_frame = self.evaluator.get_ground_truth_now_frame(unix_time)
        # [Option] ROS側でやる（Map情報・Planning結果を用いる）UC評価objectを選別
        critical_ground_truth_objects_by_ros : List[DynamicObject] = custom_critical_object_filter(ground_truth_now_frame.objects)
        # 距離などでUC評価objectを選別（EvaluationManager初期化時にConfigを設定せず、関数受け渡しにすることで動的に変更可能なInterface）
        critical_object_filter_config: CriticalObjectFilterConfig = CriticalObjectFilterConfig(
            max_position_distance,
            min_position_distance,
        )
        critical_ground_truth_objects : List[DynamicObject] = self.evaluator.critical_object_filter(
            critical_ground_truth_objects_by_ros,
            critical_object_filter_config,
        )

        # 1 frameの評価
        use_case_evaluation_config = UseCaseEvaluationConfig(threshold_plane_distance)
        frame_result = self.evaluator.add_frame_result(
            unix_time,
            ground_truth_now_frame,
            predicted_objects,
            critical_ground_truth_objects,
            use_case_evaluation_config,
        )
        uc_fail_objects = frame_result.use_case_evaluation.uc_fail_objects
        logger.debug(f"metrics result {frame_result.metrics_score}")
```
