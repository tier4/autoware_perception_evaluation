## Case 1. rosbag を用いずにデータセットのみで評価を行う

- 評価 Metrics の開発などに
- poetry install

```bash
pip3 install poetry
```

- setting

```bash
git clone https://github.com/tier4/AWMLevaluation.git
cd AWMLevaluation
poetory install
```

- 実行

```bash
cd awml_evaluation
poetry run python3 -m test.lsim <DATASET_PATH1> <DATASET_PATH2> ...
```

## Case 2. Lsim 側に組み込む

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

### 実装

- <https://github.com/tier4/AWMLevaluation/blob/main/awml_evaluation/test/lsim.py> を参照

## Case 3. local で model の評価を行う

TBD
