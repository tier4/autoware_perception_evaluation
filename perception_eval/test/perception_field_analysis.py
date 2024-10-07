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

import numpy as np
from perception_eval.tool import PerceptionAnalyzer3DField
from perception_eval.tool import PerceptionFieldAxis
from perception_eval.tool import PerceptionFieldXY
from perception_eval.visualization.perception_visualizer3dfield import PerceptionFieldPlot
from perception_eval.visualization.perception_visualizer3dfield import PerceptionFieldPlots


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

        if analyzer.num_scene == 0:
            raise ValueError("No frame results were added from "
                             f"folder {result_root_directory}."
                             "Check if the folder has been evaluated "
                             "(aka contains scene_result.pkl files.)")

        # Add columns
        analyzer.add_additional_column()
        analyzer.add_error_columns()

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

        # Define axes
        # cartesian coordinate position
        grid_axis_xy: np.ndarray = np.array([-90, -65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65, 90])
        axis_x: PerceptionFieldAxis = PerceptionFieldAxis(quantity_type="length", data_label="x")
        axis_y: PerceptionFieldAxis = PerceptionFieldAxis(quantity_type="length", data_label="y")
        axis_x.set_grid_axis(grid_axis_xy)
        axis_y.set_grid_axis(grid_axis_xy)
        # plane distance
        axis_dist: PerceptionFieldAxis = PerceptionFieldAxis(quantity_type="length", data_label="dist", name="Distance")
        grid_axis_dist: np.ndarray = np.arange(0, 105, 10)
        axis_dist.set_grid_axis(grid_axis_dist)
        axis_dist.plot_range = (0.0, 110.0)
        # position error
        axis_error_delta: PerceptionFieldAxis = PerceptionFieldAxis(
            quantity_type="length", data_label="error_delta", name="Position Error"
        )
        grid_axis_error: np.ndarray = np.arange(0, 8.0, 0.5)
        axis_error_delta.set_grid_axis(grid_axis_error)
        axis_error_delta.plot_range = (0.0, 6.0)
        axis_error_delta.plot_aspect_ratio = 1.0
        # visual heading angle
        axis_heading: PerceptionFieldAxis = PerceptionFieldAxis(
            quantity_type="angle", data_label="visual_heading", name="Heading"
        )
        # yaw error
        axis_error_yaw: PerceptionFieldAxis = PerceptionFieldAxis(
            quantity_type="angle", data_label="error_yaw", name="Yaw Error"
        )
        # none
        axis_none: PerceptionFieldAxis = PerceptionFieldAxis(quantity_type="none", data_label="none", name="None")

        plots: PerceptionFieldPlots = PerceptionFieldPlots(plot_dir)

        # 2D xy grid
        # Analysis
        error_field, _ = analyzer.analyze_xy(axis_x, axis_y, **kwargs)
        # Visualization
        plots.plot_field_basics(error_field, prefix="XY")

        # distance-visual_heading grid
        # Analysis
        error_field_dist_heading, uncertainty_field_dist_heading = analyzer.analyze_xy(
            axis_dist, axis_heading, **kwargs
        )
        # Visualization
        plots.plot_field_basics(error_field_dist_heading, prefix="dist_heading")
        plots.plot_field_basics(uncertainty_field_dist_heading, prefix="dist_heading", is_uncertainty=True)

        # Save plots, show and close
        plots.save()
        if self._show:
            plots.show()
        plots.close()

        # Individual analysis
        prefix: str = ""

        # Dist-error grid
        error_field_range, _ = analyzer.analyze_xy(axis_dist, axis_error_delta, **kwargs)
        field = error_field_range
        numb_log: np.ndarray

        if field.has_any_error_data:
            prefix = "dist_delta-error"
            numb = field.num
            numb[numb == 0] = np.nan
            numb_log = np.log10(field.num)
            # plot
            plots.plot_custom_field(field, numb_log, prefix + "_" + "numb_log", "log10 of samples [-]", vmin=0)

        # heading-yaw_error grid
        error_field_yaw_error, _ = analyzer.analyze_xy(axis_heading, axis_error_yaw, **kwargs)
        field = error_field_yaw_error

        if field.has_any_error_data:
            prefix = "yaw_error"
            numb = field.num
            numb[numb == 0] = np.nan
            numb_log = np.log10(field.num)
            # plot
            plots.plot_custom_field(field, numb_log, prefix + "_" + "numb_log", "log10 of samples [-]", vmin=0)

        # Single axis analysis
        # distance_heading grid
        error_field_dist_1d, uncertainty_field_dist_1d = analyzer.analyze_xy(axis_dist, axis_none, **kwargs)
        plots.plot_axis_basic(error_field_dist_1d, prefix="dist_1D")
        plots.plot_axis_basic(uncertainty_field_dist_1d, prefix="dist_1D", is_uncertainty=True)

        # Save plots, show and close
        plots.save()
        if self._show:
            plots.show()
        plots.close()


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
