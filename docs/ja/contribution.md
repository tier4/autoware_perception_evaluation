## coding rule

- See [detail](../ja/contribution.md)

### Static code analysis using pre-commit

- installation

```bash
pip3 install pre-commit
pre-commit install
```

- formatting

```bash
pre-commit run -a
```

## Test

### Set dataset

- Add scenario.yaml to TIER IV dataset

```yaml
├── AWMLevaluation
│   ├── awml_evaluation
├── dataset_3d
│   ├── nuscenes
│   │   └── v1.0-mini
│   └── tier4
│       ├── 202109_3d_cuboid_v2_0_1_sample
│       │   └── 60f2669b1070d0002dcdd475
│       │       ├── annotation
│       │       ├── data
│       │       ├── maps
│       │       ├── ros2bag
│       │       ├── scenario.yaml
│       │       └── status.json
```

- tips : `202109_3d_cuboid_v2_0_1_sample/60f2669b1070d0002dcdd475` is [here](https://drive.google.com/drive/u/0/folders/1WuMBQld6VPnTZ8ZVLhAJW39R_9B-w2cy) as of 2022/06/16

### Test for merge to develop branch

- unit test
  - prerequisite : ROS

```bash
cd awml_evaluation
poetry run python3 -m unittest -v
poetry run python3 -m pytest test/visualization/
```

- API test
  - e.g. `<DATASET_PATH1>` : `/dataset_3d/tier4/202109_3d_cuboid_v2_0_1_sample/60f2669b1070d0002dcdd475/`

```bash
cd awml_evaluation
poetry run python3 -m test.sensing_lsim <DATASET_PATH1> <DATASET_PATH2> ...
poetry run python3 -m test.perception_lsim <DATASET_PATH1> <DATASET_PATH2> ...
poetry run python3 -m test.eda <DATASET_PATH1> ...
```

### Test for merge to main branch

- Fix [logsim code](https://github.com/tier4/logsim/blob/ros2/logsim/scripts/perception_evaluator_node.py) for release
- Install and build Autoware
- Set dataset for logsim
  - rename ros2bag to input_bag

```yaml
├── lsim
│   ├── input
│   │   └── sample
│   │       ├── annotation
│   │       ├── data
│   │       ├── input_bag
│   │       ├── maps
│   │       ├── scenario.yaml
│   │       └── status.json
```

- Check rosbag replay and ros2 topic echo

```bash
ros2 launch autoware_launch logging_simulator.launch.xml map_path:=$HOME/local/results/input/maps vehicle_model:=gsm8 sensor_model:=aip_x2 control:=false planning:=false vehicle_id:=ps1/20210620/CAL_000015
ros2 bag play ./input_bag/ --clock 200 --rate 0.1
ros2 topic echo /sensing/lidar/concatenated/pointcloud --no-arr
```

- set cli

```bash
pipx install git+ssh://git@github.com/tier4/logsim.git
```

- set cli setting to HOME/.logsim.config.toml

```yaml
[default]
data_directory = "$HOME/autoware/lsim/input"
output_directory = "$HOME/autoware/lsim/output"
proj_directory = "$HOME/autoware/autoware.proj.gsm8"
```

- set scenario.yaml

```yaml
ScenarioFormatVersion: 2.2.0
ScenarioName: perception_sample
ScenarioDescription: perception_sample

SensorModel: aip_x2
VehicleModel: gsm8
VehicleId: ps1/20210620/CAL_000015

LocalMapPath: $HOME/autoware/lsim/input/sample/maps

Annotation: null

Evaluation:
  UseCaseName: perception
  UseCaseFormatVersion: 0.1.0
  Config: null # to set awml_evaluator config
  Conditions:
    PassRate: 99.0
```

## Branch rule

### Branch

- Main branch: logsim に使われている branch
  - merge するたびに version が上がる(ex. 1.0 -> 1.1)
  - 基本的に develop ブランチ以外を merge しない
- Develop branch: 開発用 branch
  - topic_branch からの pull_request はここに受ける
- topic_branch
  - develop に取り込んだのち削除
  - 開発速度優先で細かく commit しコミットメッセージもある程度適当でも良い
  - feature/fix などの prefix をつける

### Merge rule

- topic branch-> develop branch
  - Squash and merge で PR ごとの commit にする
- develop branch -> master branch
  - topic branch release/v1.x を作って結合作業を行う
    - version を書き換える
      - pyproject.toml
      - package.xml
    - develop に merge する
  - 手動 merge
    - merge commit を作って PR ごとの commit を維持する
  - アプリケーションとの結合を行うリリース作業に当たる

## library の構成について

- ros package 化
  - 使われる時は ros package として include される
    - autoware_utils と同様な感じになる
  - package.xml を追加する
  - 縛り：/awml_evaluation/awml_evaluation 以下に全ての code を入れる(ROS パッケージとして import するため)
