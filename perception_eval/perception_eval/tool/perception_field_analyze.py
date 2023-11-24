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

import matplotlib.pyplot as plt
import simplejson as json
import numpy as np


class PerceptionFieldPlot:
    def __init__(self, figname: str) -> None:
        self.name: str = figname
        self.figure = plt.figure(self.name, figsize=(10, 8))
        self.ax = self.figure.add_subplot(111)

        self.ax.set_aspect("equal")

    def pcolormesh(self, x, y, z, **kwargs):
        self.cs = self.ax.pcolormesh(x, y, z, shading="nearest", **kwargs)
        self.cbar = self.figure.colorbar(self.cs)

    def contourf(self, x, y, z, **kwargs):
        self.cs = self.ax.contourf(x, y, z, **kwargs)
        self.ax.contour(self.cs, colors="k")
        self.cbar = self.figure.colorbar(self.cs)

    def setAxis1D(self, field: PerceptionFieldXY, value: np.ndarray) -> None:
        if np.all(np.isnan(value)):
            return
        value_range = [np.nanmin(value), np.nanmax(value)]
        if value_range[0] == value_range[1]:
            value_range[0] -= 1
            value_range[1] += 1

        self.ax.set_xlabel(field.axis_x.getTitle())

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

    def plotScatter(self, x, y, **kwargs) -> None:
        self.cs = self.ax.scatter(x, y, **kwargs)

    def plotScatter3D(self, x, y, z, **kwargs) -> None:
        self.ax.clear()
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.cs = self.ax.scatter(x, y, z, **kwargs)

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
        analyzer.addErrorColumns()

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

        # # for debug
        # print(analyzer.df)
        # print(analyzer.df.columns)

    def analyseAndVisualize(
        self, analyzer: PerceptionAnalyzer3DField, subfolder: str, **kwargs
    ) -> None:
        plot_dir: str = Path(self._plot_dir, subfolder).as_posix()
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        # Define axes
        # cartesian coordinate position
        grid_axis_xy: np.ndarray = np.array(
            [-90, -65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65, 90]
        )
        axis_x: PerceptionFieldAxis = PerceptionFieldAxis(type="length", data_label="x")
        axis_y: PerceptionFieldAxis = PerceptionFieldAxis(type="length", data_label="y")
        axis_x.setGridAxis(grid_axis_xy)
        axis_y.setGridAxis(grid_axis_xy)
        # plane distance
        axis_dist: PerceptionFieldAxis = PerceptionFieldAxis(
            type="length", data_label="dist", name="Distance"
        )
        grid_axis_dist: np.ndarray = np.arange(0, 105, 10)
        axis_dist.setGridAxis(grid_axis_dist)
        axis_dist.plot_range = (0.0, 110.0)
        # position error
        axis_error_delta: PerceptionFieldAxis = PerceptionFieldAxis(
            type="length", data_label="error_delta", name="Position Error"
        )
        grid_axis_error: np.ndarray = np.arange(0, 8.0, 0.5)
        axis_error_delta.setGridAxis(grid_axis_error)
        axis_error_delta.plot_range = (0.0, 6.0)
        axis_error_delta.plot_aspect_ratio = 1.0
        # visual heading angle
        axis_heding: PerceptionFieldAxis = PerceptionFieldAxis(
            type="angle", data_label="visual_heading", name="Heading"
        )
        # yaw error
        axis_error_yaw: PerceptionFieldAxis = PerceptionFieldAxis(
            type="angle", data_label="error_yaw", name="Yaw Error"
        )
        # none
        axis_none: PerceptionFieldAxis = PerceptionFieldAxis(
            type="none", data_label="none", name="None"
        )

        # 2D xy grid
        # Analysis
        error_field, _ = analyzer.analyzeXY(axis_x, axis_y, **kwargs)
        # Visualization
        self.visualize(error_field, prefix="XY", save_dir=plot_dir)

        # distance-visual_heading grid
        # Analysis
        error_field_dist_heading, _ = analyzer.analyzeXY(axis_dist, axis_heding, **kwargs)
        # Visualization
        self.visualize(error_field_dist_heading, prefix="dist_heading", save_dir=plot_dir)

        # Individual analysis
        figures: list[PerceptionFieldPlot] = []
        prefix: str = ""

        # Dist-error grid
        error_field_range, _ = analyzer.analyzeXY(axis_dist, axis_error_delta, **kwargs)
        field = error_field_range
        numb_log: np.ndarray

        if field.has_any_error_data == True:
            prefix = "dist_delta-error"
            numb = field.num
            numb[numb == 0] = np.nan
            numb_log = np.log10(field.num)
            # plot
            figures.append(PerceptionFieldPlot(prefix + "_" + "numb_log"))
            figures[-1].plotMeshMap(field, numb_log, vmin=0)
            figures[-1].setAxes(field)

        # heading-yaw_error grid
        error_field_yaw_error, _ = analyzer.analyzeXY(axis_heding, axis_error_yaw, **kwargs)
        field = error_field_yaw_error

        if field.has_any_error_data == True:
            prefix = "yaw_error"
            numb = field.num
            numb[numb == 0] = np.nan
            numb_log = np.log10(field.num)
            # plot
            figures.append(PerceptionFieldPlot(prefix + "_" + "numb_log"))
            figures[-1].plotMeshMap(field, numb_log, vmin=0)
            figures[-1].setAxes(field)

        # 1D analysis
        # distance_heading grid
        prefix = "dist_1D"

        error_field_dist_1d, _ = analyzer.analyzeXY(axis_dist, axis_none, **kwargs)
        field = error_field_dist_1d

        figures.append(PerceptionFieldPlot(prefix + "_" + "numb"))
        figures[-1].ax.scatter(field.dist, field.num, marker="x", c="r", s=10)
        figures[-1].setAxis1D(field, field.num)

        figures.append(PerceptionFieldPlot(prefix + "_" + "rates"))
        figures[-1].ax.scatter(field.dist, field.ratio_tp, marker="o", c="b", s=20)
        figures[-1].ax.scatter(field.dist, field.ratio_fn, marker="x", c="r", s=20)
        figures[-1].ax.scatter(field.dist, field.ratio_fp, marker="^", c="g", s=20)
        figures[-1].setAxis1D(field, field.ratio_tp)
        figures[-1].ax.set_ylim([0, 1])
        figures[-1].ax.set_aspect(10.0 / 0.2)

        if field.has_any_error_data == True:
            figures.append(PerceptionFieldPlot(prefix + "_" + "error_delta_bar"))
            figures[-1].ax.set_ylim([-1, 5])
            figures[-1].ax.errorbar(
                field.dist.flatten(),
                field.error_delta_mean.flatten(),
                yerr=field.error_delta_std.flatten(),
                marker="x",
                c="r",
            )
            figures[-1].setAxis1D(field, field.error_delta_mean)
            figures[-1].ax.set_aspect(10.0 / 1)
        else:
            print("No TP data, nothing for error analysis")

        # all data analysis
        analyzer.analyzeAll(**kwargs)
        prefix = "all_points"

        figures.append(PerceptionFieldPlot(prefix + "_" + "dist_diff"))
        figures[-1].plotScatter(analyzer.data_pair[:, DataTableIdx.DIST], analyzer.data_pair[:, DataTableIdx.D_DIST])

        figures.append(PerceptionFieldPlot(prefix + "_" + "azimuth_diff"))
        figures[-1].plotScatter(analyzer.data_pair[:, DataTableIdx.AZIMUTH], analyzer.data_pair[:, DataTableIdx.D_AZIMUTH])

        azimuth_error: np.ndarray = (analyzer.data_pair[:, DataTableIdx.D_AZIMUTH] - analyzer.data_pair[:, DataTableIdx.AZIMUTH])
        azimuth_error[azimuth_error > np.pi] -= 2 * np.pi
        azimuth_error[azimuth_error < -np.pi] += 2 * np.pi
        azimuth_dist_error: np.ndarray = azimuth_error * analyzer.data_pair[:, DataTableIdx.DIST]
        figures.append(PerceptionFieldPlot(prefix + "_" + "dist_latitudinal_dist_error"))
        figures[-1].plotScatter(analyzer.data_pair[:, DataTableIdx.DIST], 
                                azimuth_dist_error)
        
        dist_error = analyzer.data_pair[:, DataTableIdx.D_DIST] - analyzer.data_pair[:, DataTableIdx.DIST]
        figures.append(PerceptionFieldPlot(prefix + "_" + "X_Y_dist_error"))
        figures[-1].plotScatter3D(analyzer.data_pair[:, DataTableIdx.X],
                                  analyzer.data_pair[:, DataTableIdx.Y],
                                  dist_error)
        figures[-1].ax.set_xlabel("X")
        figures[-1].ax.set_ylabel("Y")
        figures[-1].ax.set_zlabel("dist error")

        # Save plots
        for fig in figures:
            fig.figure.savefig(Path(plot_dir, fig.name + ".png"))

        # Show plots
        if self._show:
            plt.show()

        plt.close("all")
        

    def visualize(self, field: PerceptionFieldXY, prefix: str, save_dir: str) -> None:
        # Check if there is no TP data
        is_pair_data: bool = field.num_pair[np.isnan(field.num_pair) != False].sum() != 0

        # Preprocess
        mask_layer: np.ndarray = np.zeros(np.shape(field.num_pair), dtype=np.bool_)
        mask_layer[field.num_pair == 0] = True
        field.num_pair[mask_layer] = np.nan

        # Plot
        figures = []

        # Number of data
        figures.append(PerceptionFieldPlot(prefix + "_" + "num"))
        figures[-1].plotMeshMap(field, field.num, vmin=0)
        figures[-1].setAxes(field)

        # True positive rate
        figures.append(PerceptionFieldPlot(prefix + "_" + "ratio_tp"))
        figures[-1].plotMeshMap(field, field.ratio_tp, vmin=0, vmax=1)
        figures[-1].setAxes(field)

        # False positive rate
        figures.append(PerceptionFieldPlot(prefix + "_" + "ratio_fp"))
        figures[-1].plotMeshMap(field, field.ratio_fp, vmin=0, vmax=1)
        figures[-1].setAxes(field)

        # False negative rate
        figures.append(PerceptionFieldPlot(prefix + "_" + "ratio_fn"))
        figures[-1].plotMeshMap(field, field.ratio_fn, vmin=0, vmax=1)
        figures[-1].setAxes(field)

        # Position error
        if field.has_any_error_data:
            figures.append(PerceptionFieldPlot(prefix + "_" + "delta_mean_mesh"))
            vmax = 1
            if np.all(np.isnan(field.error_delta_mean)) != True:
                vmax = np.nanmax(field.error_delta_mean)
            figures[-1].plotMeshMap(field, field.error_delta_mean, vmin=0, vmax=vmax)

            # mean positions of each grid
            if hasattr(field, field.axis_x.data_label):
                x_mean_plot = getattr(field, field.axis_x.data_label) * field.axis_x.plot_scale
            else:
                x_mean_plot = field.mesh_center_x * field.axis_x.plot_scale
            if hasattr(field, field.axis_y.data_label):
                y_mean_plot = getattr(field, field.axis_y.data_label) * field.axis_y.plot_scale
            else:
                y_mean_plot = field.mesh_center_y * field.axis_y.plot_scale

            cs = figures[-1].ax.scatter(x_mean_plot, y_mean_plot, marker="+", c="r", s=10)
            figures[-1].setAxes(field)
        else:
            print("No TP data, nothing for error analysis")

        # Save result
        for fig in figures:
            fig.figure.savefig(Path(save_dir, fig.name + ".png"))


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
        "--show-plot",
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
