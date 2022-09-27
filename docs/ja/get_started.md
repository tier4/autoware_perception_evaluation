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
  git clone https://github.com/tier4/autoware_perception_evaluation.git
  cd autoware_perception_evaluation
  poetry install
  ```

  - 暫定的な措置として，EDA ツール用のライブラリを pip で install

    ```bash
    poetry shell
    pip install -r requirements.txt
    ```

  - 実行

  ```bash
  cd perception_eval
  poetry run python3 -m test.sensing_lsim <DATASET_PATH1> <DATASET_PATH2> ...
  poetry run python3 -m test.perception_lsim <DATASET_PATH1> <DATASET_PATH2> ...
  ```

- ライブラリとして使用

  ```bash
  git clone https://github.com/tier4/autoware_perception_evaluation.git
  cd <YOUR_PROJECT>
  poetry add <RELATIVE_PATH_TO_autoware_perception_evaluation> # または pip install -e <RELATIVE_PATH_TO_autoware_perception_evaluation>
  ```

  -　使用例

  ```bash
  ~/workspace/test_project $ poetry run python3
  Python 3.8.10 (default, Mar 15 2022, 12:22:08)
  [GCC 9.4.0] on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> from perception_eval.common.dataset import load_all_datasets
  >>> from perception_eval.perception_evaluation_manager import PerceptionEvaluationManager
  >>> from perception_eval.sensing_evaluation_manager import SensingEvaluationManager
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
  simulator/autoware_perception_evaluation:
    type: git
    url: git@github.com:tier4/autoware_perception_evaluation.git
    version: main
```

## 実装

- ライブラリ or ROS パッケージとして使う場合
  - Perception: [perception_eval/test/perception_lsim.py](../../perception_eval/test/perception_lsim.py)
  - Sensing: [perception_eval/test/sensing_lsim.py](../../perception_eval/test/sensing_lsim.py)
