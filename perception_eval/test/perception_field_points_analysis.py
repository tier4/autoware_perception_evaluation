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
import os
from os.path import expandvars
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from perception_eval.tool import DataTableIdx
from perception_eval.tool import PerceptionAnalyzer3DField
from perception_eval.visualization.perception_visualizer3dfield import PerceptionFieldPlot


class PerceptionLoadDatabaseResult:
    def __init__(self, result_root_directory: str, scenario_path: str, show: bool = False) -> None:
        self._result_root_directory: str = result_root_directory
        self._plot_dir: str = Path(result_root_directory, "plot").as_posix()
        self._show: bool = show

        if not os.path.isdir(self._plot_dir):
            os.makedirs(self._plot_dir)

        # Initialize
        analyzer: PerceptionAnalyzer3DField = PerceptionAnalyzer3DField.from_scenario(
            result_root_directory,
            scenario_path,
        )

        # Load files
        pickle_file_paths = Path(result_root_directory).glob("**/scene_result.pkl")
        for filepath in pickle_file_paths:
            analyzer.add_from_pkl(filepath.as_posix())

        # Add columns
        analyzer.add_additional_column()

        # Analyze and visualize, for each label group
        label_lists: dict = {}
        label_lists["All"] = None
        label_lists["Vehicle"] = ["car", "truck"]
        label_lists["VRU"] = ["pedestrian", "bicycle"]
        label_lists["Motorbike"] = ["motorbike"]

        for label_group, labels in label_lists.items():
            print('Analyzing label group: {}, label list of "{}" '.format(label_group, labels))
            self.analyse_and_visualize(analyzer, subfolder=label_group, label=labels)
            print("Done")

    def analyse_and_visualize(self, analyzer: PerceptionAnalyzer3DField, subfolder: str, **kwargs) -> None:
        plot_dir: str = Path(self._plot_dir, subfolder).as_posix()
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        # all objects analysis
        figures: list[PerceptionFieldPlot] = []
        analyzer.analyze_points(**kwargs)

        # true positives
        prefix = "points_tp"
        table_gt = analyzer.data_tp_gt
        table_est = analyzer.data_tp_est

        figures.append(PerceptionFieldPlot(prefix + "_" + "dist_diff", "Distance error [m]"))
        figures[-1].plot_scatter(table_gt[:, DataTableIdx.DIST], table_est[:, DataTableIdx.DIST])
        figures[-1].ax.set_xlabel("GT Distance [m]")
        figures[-1].ax.set_ylabel("Est Distance [m]")

        figures.append(PerceptionFieldPlot(prefix + "_" + "azimuth_diff", "Azimuth error [rad]"))
        figures[-1].plot_scatter(table_gt[:, DataTableIdx.AZIMUTH], table_est[:, DataTableIdx.AZIMUTH])
        figures[-1].ax.set_xlabel("GT Azimuth [rad]")
        figures[-1].ax.set_ylabel("Est Azimuth [rad]")

        azimuth_error: np.ndarray = table_est[:, DataTableIdx.AZIMUTH] - table_gt[:, DataTableIdx.AZIMUTH]
        azimuth_error[azimuth_error > np.pi] -= 2 * np.pi
        azimuth_error[azimuth_error < -np.pi] += 2 * np.pi
        azimuth_dist_error: np.ndarray = azimuth_error * table_gt[:, DataTableIdx.DIST]
        figures.append(
            PerceptionFieldPlot(prefix + "_" + "dist_latitudinal_position_error", "Latitudinal position error [m]")
        )
        figures[-1].plot_scatter(table_gt[:, DataTableIdx.DIST], azimuth_dist_error)
        figures[-1].ax.set_xlabel("GT Distance [m]")
        figures[-1].ax.set_ylabel("Latitudinal position error [m]")

        dist_error = table_est[:, DataTableIdx.DIST] - table_gt[:, DataTableIdx.DIST]
        figures.append(PerceptionFieldPlot(prefix + "_" + "TP_XY_dist_error", "Position error [m]"))
        figures[-1].plot_scatter_3d(table_gt[:, DataTableIdx.X], table_gt[:, DataTableIdx.Y], dist_error)
        figures[-1].ax.set_xlabel("X [m]")
        figures[-1].ax.set_ylabel("Y [m]")

        # false negatives
        table = analyzer.data_fn
        figures.append(PerceptionFieldPlot(prefix + "_" + "FN_XY_width", "Width [m]"))
        figures[-1].plot_scatter_3d(table[:, DataTableIdx.X], table[:, DataTableIdx.Y], table[:, DataTableIdx.WIDTH])
        figures[-1].ax.set_xlabel("X [m]")
        figures[-1].ax.set_ylabel("Y [m]")

        # Save plots
        for fig in figures:
            fig.figure.savefig(Path(plot_dir, fig.name + ".png"))

        # Show plots
        if self._show:
            plt.show()

        plt.close("all")


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
    parser.add_argument(
        "--show",
        action="store_true",
        dest="show",
        help="show plots",
    )
    args = parser.parse_args()
    PerceptionLoadDatabaseResult(
        expandvars(args.result_root_directory),
        expandvars(args.scenario_path),
        args.show,
    )


if __name__ == "__main__":
    main()
