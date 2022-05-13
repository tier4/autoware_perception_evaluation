# Get started

## Case 1. rosbag を用いずにデータセットのみで評価を行う

- ソースコードを使用 or ライブラリとして使用
  - 評価 Metrics の開発/利用などに

### install

- poetry install

```bash
pip3 install poetry
```

- ソースコードを使用

  ```bash
  git clone https://github.com/tier4/AWMLevaluation.git
  cd AWMLevaluation
  poetry install
  ```

  - 実行

  ```bash
  cd awml_evaluation
  poetry run python3 -m test.lsim <DATASET_PATH1> <DATASET_PATH2> ...
  ```

- ライブラリとして使用

  ```bash
  git clone https://github.com/tier4/AWMLevaluation.git
  cd <YOUR_PROJECT>
  poetry add <RELATIVE_PATH_TO_AWMLevaluation> # または pip install -e <RELATIVE_PATH_TO_AWMLevaluation>
  ```

  -　使用例

  ```bash
  ~/workspace/test_project $ poetry run python3
  Python 3.8.10 (default, Mar 15 2022, 12:22:08)
  [GCC 9.4.0] on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> from awml_evaluation.common.dataset import load_all_datasets
  >>> from awml_evaluation.perception_evaluation_manager import PerceptionEvaluationManager
  >>> from awml_evaluation.sensing_evaluation_manager import SensingEvaluationManager
  ```

## Case 2. Lsim 側に組み込む

- ROS パッケージとして使用する

### install

- pip, submodule 等は使わない
- repos に追加
  - 評価地点に関しては hash で管理する

```yaml
repositories:
  # autoware
  simulator/AWMLevaluation:
    type: git
    url: git@github.com/tier4/AWMLevaluation.git
    version: main
```

## 実装

- ライブラリ or ROS パッケージとして使う場合
  - <https://github.com/tier4/AWMLevaluation/blob/main/awml_evaluation/test/lsim.py> を参照

## その他の case

- Perception: <https://github.com/tier4/AWMLevaluation/blob/main/docs/perception/perception_other_cases.md>を参照
- Sensing: <https://github.com/tier4/AWMLevaluation/blob/main/docs/sensing/sensing_other_cases.md>を参照
