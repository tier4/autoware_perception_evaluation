# Get started

## Case 1. Evaluate only with your dataset without rosbag

- With source code or use this package as library.
  - For developing or just using metrics

### install

- Install poetry

```bash
pip3 install poetry
```

- Use source code

  ```bash
  git clone https://github.com/tier4/autoware_perception_evaluation.git
  cd autoware_perception_evaluation
  poetry install
  ```

  - Use pip install for the dependencies of EDA tool

    ```bash
    poetry shell
    pip install -r requirements.txt
    ```

  - Run

  ```bash
  cd perception_eval
  poetry run python3 -m test.sensing_lsim <DATASET_PATH1> <DATASET_PATH2> ...
  poetry run python3 -m test.perception_lsim <DATASET_PATH1> <DATASET_PATH2> ...
  ```

- Use as a library

  ```bash
  git clone https://github.com/tier4/autoware_perception_evaluation.git
  cd <YOUR_PROJECT>
  poetry add <RELATIVE_PATH_TO_autoware_perception_evaluation> # or pip install -e <RELATIVE_PATH_TO_autoware_perception_evaluation>
  ```

  - Example of use

  ```bash
  ~/workspace/test_project $ poetry run python3
  Python 3.8.10 (default, Mar 15 2022, 12:22:08)
  [GCC 9.4.0] on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> from perception_eval.common.dataset import load_all_datasets
  >>> from perception_eval.perception_evaluation_manager import PerceptionEvaluationManager
  >>> from perception_eval.sensing_evaluation_manager import SensingEvaluationManager
  ```

## Case 2. Used in driving_log_replayer

- Use as ROS package

### install

- Do not use `pip` or `submodule`
- Add to .repos
  - Manage with tag

```yaml
repositories:
  # autoware
  simulator/autoware_perception_evaluation:
    type: git
    url: git@github.com:tier4/autoware_perception_evaluation.git
    version: main
```

## Example of implementation

- In case of using as a library or ROS package
  - Perception: [perception_eval/test/perception_lsim.py](../../perception_eval/test/perception_lsim.py)
  - Sensing: [perception_eval/test/sensing_lsim.py](../../perception_eval/test/sensing_lsim.py)
