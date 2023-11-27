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

from perception_eval.tool import PerceptionAnalyzer3DField
from perception_eval.tool import PerceptionFieldXY
from perception_eval.tool import PerceptionFieldAxis
from perception_eval.tool import DataTableIdx

from perception_eval.visualization.perception_visualizer3dfield import PerceptionFieldPlot

import matplotlib.pyplot as plt
import numpy as np


class PerceptionFieldPlot:
    def __init__(self, figname: str, value: str = "Value []") -> None:
        self.name: str = figname
        self.figure = plt.figure(self.name, figsize=(10, 8))
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect("equal")
        self.value: str = value

    def pcolormesh(self, x, y, z, **kwargs):
        self.cs = self.ax.pcolormesh(x, y, z, shading="nearest", **kwargs)
        self.cbar = self.figure.colorbar(self.cs)
        self.cbar.set_label(self.value)

    def contourf(self, x, y, z, **kwargs):
        self.cs = self.ax.contourf(x, y, z, **kwargs)
        self.ax.contour(self.cs, colors="k")
        self.cbar = self.figure.colorbar(self.cs)
        self.cbar.set_label(self.value)

    def setAxis1D(self, field: PerceptionFieldXY, value: np.ndarray) -> None:
        if np.all(np.isnan(value)):
            return
        value_range = [np.nanmin(value), np.nanmax(value)]
        if value_range[0] == value_range[1]:
            value_range[0] -= 1
            value_range[1] += 1

        self.ax.set_xlabel(field.axis_x.getTitle())
        self.ax.set_ylabel(self.value)

        value_scale = (value_range[1] - value_range[0]) / 10.0
        value_plot_margin = value_scale / 2.0  # 5% margin
        self.ax.set_aspect(field.axis_x.plot_aspect_ratio / value_scale)
        self.ax.set_xlim(field.axis_x.plot_range)
        self.ax.set_ylim(value_range[0] - value_plot_margin, value_range[1] + value_plot_margin)
        self.ax.grid(c="k", ls="-", alpha=0.3)
        self.ax.set_xticks(field.axis_x.grid_axis * field.axis_x.plot_scale)

    def setAxes(self, field: PerceptionFieldXY) -> None:
        self.ax.set_xlabel(field.axis_x.getTitle())
        self.ax.set_ylabel(field.axis_y.getTitle())
        self.ax.set_aspect(field.axis_x.plot_aspect_ratio / field.axis_y.plot_aspect_ratio)
        self.ax.set_xlim(field.axis_x.plot_range)
        self.ax.set_ylim(field.axis_y.plot_range)
        self.ax.grid(c="k", ls="-", alpha=0.3)
        self.ax.set_xticks(field.axis_x.grid_axis * field.axis_x.plot_scale)
        self.ax.set_yticks(field.axis_y.grid_axis * field.axis_y.plot_scale)

    def plotMeshMap(self, field: PerceptionFieldXY, valuemap: np.ndarray, **kwargs) -> None:
        x = field.mesh_center_x * field.axis_x.plot_scale
        y = field.mesh_center_y * field.axis_y.plot_scale
        z = valuemap
        self.cs = self.ax.pcolormesh(x, y, z, shading="nearest", **kwargs)
        self.cbar = self.figure.colorbar(self.cs)
        self.cbar.set_label(self.value)

    def plotScatter(self, x, y, **kwargs) -> None:
        self.cs = self.ax.scatter(x, y, **kwargs)

    def plotScatter3D(self, x, y, z, **kwargs) -> None:
        self.ax.remove()
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.cs = self.ax.scatter(x, y, z, **kwargs)
        # enlarge figure size
        self.figure.set_size_inches(14, 10)
        self.ax.set_zlabel(self.value)


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
        analyzer.addAdditionalColumn()

        # Analyze and visualize, for each label group
        label_lists: dict = {}
        label_lists["All"] = None
        label_lists["Vehicle"] = ["car", "truck"]
        label_lists["VRU"] = ["pedestrian", "bicycle"]
        label_lists["Motorbike"] = ["motorbike"]

        for label_group, labels in label_lists.items():
            print('Analyzing label group: {}, label list of "{}" '.format(label_group, labels))
            self.analyseAndVisualize(analyzer, subfolder=label_group, label=labels)
            print("Done")

    def analyseAndVisualize(
        self, analyzer: PerceptionAnalyzer3DField, subfolder: str, **kwargs
    ) -> None:
        plot_dir: str = Path(self._plot_dir, subfolder).as_posix()
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        # all objects analysis
        figures: list[PerceptionFieldPlot] = []
        analyzer.analyzePoints(**kwargs)

        # true positives
        prefix = "points_tp"
        table_gt = analyzer.data_tp_gt
        table_est = analyzer.data_tp_est

        figures.append(PerceptionFieldPlot(prefix + "_" + "dist_diff", "Distance error [m]"))
        figures[-1].plotScatter(table_gt[:, DataTableIdx.DIST], table_est[:, DataTableIdx.DIST])
        figures[-1].ax.set_xlabel("GT Distance [m]")
        figures[-1].ax.set_ylabel("Est Distance [m]")

        figures.append(PerceptionFieldPlot(prefix + "_" + "azimuth_diff", "Azimuth error [rad]"))
        figures[-1].plotScatter(
            table_gt[:, DataTableIdx.AZIMUTH], table_est[:, DataTableIdx.AZIMUTH]
        )
        figures[-1].ax.set_xlabel("GT Azimuth [rad]")
        figures[-1].ax.set_ylabel("Est Azimuth [rad]")

        azimuth_error: np.ndarray = (
            table_est[:, DataTableIdx.AZIMUTH] - table_gt[:, DataTableIdx.AZIMUTH]
        )
        azimuth_error[azimuth_error > np.pi] -= 2 * np.pi
        azimuth_error[azimuth_error < -np.pi] += 2 * np.pi
        azimuth_dist_error: np.ndarray = azimuth_error * table_gt[:, DataTableIdx.DIST]
        figures.append(
            PerceptionFieldPlot(
                prefix + "_" + "dist_latitudinal_position_error", "Latitudinal position error [m]"
            )
        )
        figures[-1].plotScatter(table_gt[:, DataTableIdx.DIST], azimuth_dist_error)
        figures[-1].ax.set_xlabel("GT Distance [m]")
        figures[-1].ax.set_ylabel("Latitudinal position error [m]")

        dist_error = table_est[:, DataTableIdx.DIST] - table_gt[:, DataTableIdx.DIST]
        figures.append(PerceptionFieldPlot(prefix + "_" + "TP_XY_dist_error", "Position error [m]"))
        figures[-1].plotScatter3D(
            table_gt[:, DataTableIdx.X], table_gt[:, DataTableIdx.Y], dist_error
        )
        figures[-1].ax.set_xlabel("X [m]")
        figures[-1].ax.set_ylabel("Y [m]")

        # false negatives
        table = analyzer.data_fn
        figures.append(PerceptionFieldPlot(prefix + "_" + "FN_XY_width", "Width [m]"))
        figures[-1].plotScatter3D(
            table[:, DataTableIdx.X], table[:, DataTableIdx.Y], table[:, DataTableIdx.WIDTH]
        )
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
