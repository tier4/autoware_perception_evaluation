#!/usr/bin/env python3

# Copyright (c) 2023 TIER IV.inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from os.path import expandvars
from pathlib import Path

from perception_eval.tool import PerceptionAnalyzer3DField

import simplejson as json


class PerceptionLoadDatabaseResult:
    def __init__(self, result_root_directory: str, scenario_path: str) -> None:
        # Initialize
        analyzer = PerceptionAnalyzer3DField.from_scenario(
            result_root_directory,
            scenario_path,
        )

        # Load files
        pickle_file_paths = Path(result_root_directory).glob("**/scene_result.pkl")
        for filepath in pickle_file_paths:
            analyzer.add_from_pkl(filepath.as_posix())

        # Analyze
          # conditions: object class, grid type, error/uncertainty...
        score_field, error_field = analyzer.analyze()


        print(analyzer.df)


        # Plot



        # # Save result
        # result_file_path = Path(result_root_directory, "database_result.json")
        # with result_file_path.open("w") as f:
        #     json.dump(database_metrics, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--result_root_directory",
        required=True,
        help="root directory of result",
    )
    parser.add_argument(
        "-s",
        "--scenario_path",
        required=True,
        help="path of the scenario to load evaluator settings",
    )
    args = parser.parse_args()
    PerceptionLoadDatabaseResult(
        expandvars(args.result_root_directory),
        expandvars(args.scenario_path),
    )


if __name__ == "__main__":
    main()
