## coding rule

- See [detail](https://github.com/tier4/AWMLtools/blob/main/docs/development/contribution.md)

## Test
### Set dataset

- Add scenario.yaml to Tier4 dataset

```
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


### Test for merge to develop branch

- unit test

```
cd awml_evaluation
python3 -m unittest -v
```

- API test
```
cd awml_evaluation
poetry run python3 -m test.lsim
```

### Test for merge to main branch

- Fix [logsim code](https://github.com/tier4/logsim/blob/ros2/logsim/scripts/perception_evaluator_node.py) for release
- Install and build Autoware
- Set dataset for logsim
  - rename ros2bag to input_bag

```
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

```sh
ros2 launch autoware_launch logging_simulator.launch.xml map_path:=$HOME/local/results/input/maps vehicle_model:=gsm8 sensor_model:=aip_x2 control:=false planning:=false vehicle_id:=ps1/20210620/CAL_000015
ros2 bag play ./input_bag/ --clock 200 --rate 0.1
ros2 topic echo /sensing/lidar/concatenated/pointcloud --no-arr
```

- set cli

```
pipx install git+ssh://git@github.com/tier4/logsim.git
```

- set cli setting to HOME/.logsim.config.toml

```
[default]
data_directory = "$HOME/autoware/lsim/input"
output_directory = "$HOME/autoware/lsim/output"
proj_directory = "$HOME/autoware/autoware.proj.gsm8"
```

- set scenario.yaml

```yaml
ScenarioFormatVersion : 2.2.0
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

- Main branch: logsimに使われているbranch
  - mergeするたびにversionが上がる(ex. 1.0 -> 1.1)
  - 基本的にdevelopブランチ以外をmergeしない
- Develop branch: 開発用branch
  - topic_branchからのpull_requestはここに受ける
- topic_branch
  - developに取り込んだのち削除
  - 開発速度優先で細かくcommitしコミットメッセージもある程度適当でも良い
  - feature/fix などのprefixをつける

### Merge rule

- topic branch-> develop branch
  - Squash and mergeでPRごとのcommitにする
- develop branch -> master branch
  - 手動merge merge commitを作ってPRごとのcommitを維持する
  - アプリケーションとの結合を行うリリース作業に当たる
  - 何か問題が出たらtopic branch release/v1.x などを作って結合作業を行う

## library の構成について

- ros package化
  - 使われる時はros packageとしてincludeされる
    - autoware_utilsと同様な感じになる
  - package.xmlを追加する
  - 縛り：/awml_evaluation/awml_evaluation 以下に全てのcodeを入れる(ROSパッケージとしてimportするため)
