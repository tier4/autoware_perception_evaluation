## Coding rules

- Add **type hinting** like below.

  ```python
  from typing import List

  num: int = 10
  names: List[str] = ["Alice", "Bob"]
  ```

- Add **docstring** for class and function
- Use `black` for python formatter. If you are using vs-code, there is no settings you have to do.
- Use `pre-commit` before commit your updates.

### Static code analysis using pre-commit

- installation

  ```bash
  pip3 install pre-commit
  pre-commit install
  ```

- formatting

  - _NOTE_: If you have done `pre-commit install`, pre-commit run automatically when you commit changes.

  ```bash
  pre-commit run -a
  ```

## Test

### Set dataset

- Add scenario.yaml to TIER IV dataset

```yaml
├── autoware_perception_evaluation
│   ├── perception_eval
├── dataset_3d
│   ├── nuscenes
│   │   └── v1.0-mini
│   └── tier4
│       ├── t4dataset
│       │    ├── annotation
│       │    ├── data
│       │    ├── maps
│       │    ├── ros2bag
│       │    ├── scenario.yaml
│       │    └── status.json
```

### Test for merge to develop branch

- unit test
  - prerequisite : ROS

```bash
cd autoware_perception_evaluation
poetry run python3 -m unittest -v
poetry run python3 -m pytest test/visualization/
```

```bash
cd perception_eval
poetry run python3 -m test.sensing_lsim <DATASET_PATH>
poetry run python3 -m test.perception_lsim <DATASET_PATH>
poetry run python3 -m test.eda <DATASET_PATH>
```

### Test for merge to main branch

- Fix [driving_log_replayer code](https://github.com/tier4/driving_log_replayer) for release

## Branch rules

### Branch definition

- `main`: `driving_log_replayer`で使用するブランチ
  - マージごとに version を上げる(ex. v.1.0.0 -> 1.0.1)
  - develop ブランチ以外を merge しない
- `develop`: 開発用 ブランチ
  - topic_branch からの pull_request はここに受ける
- topic_branch
  - develop に取り込んだのち削除
  - feat/, fix/ などの prefix をつける

### Merge & Release rules

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
  - 縛り：perception_eval/perception_eval 以下に全ての code を入れる(ROS パッケージとして import するため)
