# Contribution

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

### Static code analysis with pre-commit

- Installation

  ```bash
  pip3 install pre-commit
  pre-commit install
  ```

- formatting

  - _NOTE_: If you have done `pre-commit install`, pre-commit run automatically when you commit changes.

  ```bash
  pre-commit run -a
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

- `main`: Branch Used in `driving_log_replayer`
  - In every merge, upgrade version (ex. v1.0.0 -> v1.0.1)
  - Do not merge except of `develop` branch
- `develop`: Branch used for development
  - receive pull requests from topic branches.
- topic branch
  - Remove after merging into `develop`
  - Add prefix like feat/ or fix/

### Merge & Release rules

- topic branch -> `develop` branch
  - Commit with Squash & Merge in every pull request
- `develop` branch -> `main` branch
  - Create topic branch named like release/v1.x
    - Upgrade version information in
      - `pyproject.toml`
      - `package.xml`
    - Merge into `develop` branch
  - Merge into `main` branch in your local
    - Create merge commit
  - Sync with the other applications
