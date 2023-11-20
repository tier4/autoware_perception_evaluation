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
from perception_eval.tool import PerceptionFieldXY
from perception_eval.tool import PerceptionFieldAxis

import matplotlib.pyplot as plt
import simplejson as json
import numpy as np

class PerceptionFieldPlot:
    def __init__(self, figname:str)-> None:
        self.name : str = figname
        self.figure = plt.figure(self.name, figsize=(10, 8))
        self.ax = self.figure.add_subplot(111)
        
        self.ax.set_aspect('equal')
    
    def pcolormesh(self, x, y, z, **kwargs):
        self.cs = self.ax.pcolormesh(x, y, z, 
                                     shading='nearest',
                                     **kwargs)
        self.cbar = self.figure.colorbar(self.cs)

    def contourf(self, x, y, z, **kwargs):
        self.cs = self.ax.contourf(x, y, z, **kwargs)
        self.ax.contour(self.cs, colors='k')
        self.cbar = self.figure.colorbar(self.cs)

    def setAxes(self, field:PerceptionAnalyzer3DField):
        self.ax.set_xlabel(field.axis_x.getTitle())
        self.ax.set_ylabel(field.axis_y.getTitle())
        self.ax.set_aspect(field.axis_x.plot_aspect_ratio / field.axis_y.plot_aspect_ratio)
        self.ax.set_xlim(field.axis_x.plot_range)
        self.ax.set_ylim(field.axis_y.plot_range)
        self.ax.grid(c='k', ls='-', alpha=0.3)
        self.ax.set_xticks(field.axis_x.grid_axis * field.axis_x.plot_scale)
        self.ax.set_yticks(field.axis_y.grid_axis * field.axis_y.plot_scale)

    def plotMeshMap(self, field:PerceptionAnalyzer3DField, valuemap:np.ndarray, **kwargs):
        x = field.mesh_center_x * field.axis_x.plot_scale
        y = field.mesh_center_y * field.axis_y.plot_scale
        z = valuemap
        self.cs = self.ax.pcolormesh(x, y, z, 
                                     shading='nearest',
                                     **kwargs)
        self.cbar = self.figure.colorbar(self.cs)

def visualize(field:PerceptionFieldXY, prefix : str, save_dir:str)->None:
    
    # Preprocess
    mask_layer :np.ndarray = np.zeros(np.shape(field.num_pair),dtype=np.bool_)
    mask_layer[field.num_pair == 0] = True

    field.num_pair[mask_layer] = np.nan
    field.ratio_tp[mask_layer] = np.nan
    field.ratio_fp[mask_layer] = np.nan
    field.ratio_fn[mask_layer] = np.nan
    field.error_delta_mean[mask_layer] = np.nan

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
    figures.append(PerceptionFieldPlot(prefix + "_" + "delta_mean_mesh"))
    figures[-1].plotMeshMap(field, field.error_delta_mean, vmin=0, vmax=np.nanmax(field.error_delta_mean))
    # mean positions of each grid
    if hasattr(field, field.axis_x.data_label):
        x_mean_plot = getattr(field,field.axis_x.data_label) * field.axis_x.plot_scale
    else:
        x_mean_plot = field.mesh_center_x * field.axis_x.plot_scale

    if hasattr(field, field.axis_y.data_label):
        y_mean_plot = getattr(field,field.axis_y.data_label) * field.axis_y.plot_scale
    else:
        y_mean_plot = field.mesh_center_y * field.axis_y.plot_scale

    cs = figures[-1].ax.scatter(x_mean_plot, y_mean_plot, marker='+',c='r', s=10)
    figures[-1].setAxes(field)

    # Save result
    plot_file_path = Path(save_dir, "plot")
        
    for fig in figures:
        fig.figure.savefig(Path(plot_file_path, fig.name + ".png"))



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

        
        # Add columns
        analyzer.addAdditionalColumn()
        analyzer.addErrorColumns()

        # Analyze
        # conditions: object class, grid type, error/uncertainty...

        # Defalult analysis : all object class, all grid type, error
        # error_field, uncertainty_field = analyzer.analyze()

        # Specific class
        # labels: car, truck, bicycle, pedestrian, motorbike
        # src/simulator/perception_eval/docs/en/perception/label.md
        # error_field, _ = analyzer.analyze(label="car") 


        # Define axes
        # cartesian coordinate position
        grid_axis_xy : np.ndarray = np.array([-90, -65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65, 90])
        axis_x : PerceptionFieldAxis = PerceptionFieldAxis(type="length",data_label="x")
        axis_y : PerceptionFieldAxis = PerceptionFieldAxis(type="length",data_label="y")
        axis_x.setGridAxis(grid_axis_xy)
        axis_y.setGridAxis(grid_axis_xy)
        # plane distance
        axis_dist : PerceptionFieldAxis = PerceptionFieldAxis(type="length", data_label="dist", name="Distance")
        grid_axis_dist : np.ndarray = np.arange(0, 105, 10)
        axis_dist.setGridAxis(grid_axis_dist)
        axis_dist.plot_range = [0, 110]
        # position error
        axis_error_delta : PerceptionFieldAxis = PerceptionFieldAxis(type="length", data_label="error_delta", name="Position Error")
        grid_axis_error : np.ndarray = np.arange(0, 8.0, 0.5)
        axis_error_delta.setGridAxis(grid_axis_error)
        axis_error_delta.plot_range = [0, 6.0]
        axis_error_delta.plot_aspect_ratio = 1.0
        # visual heading angle
        axis_heding : PerceptionFieldAxis = PerceptionFieldAxis(type="angle", data_label="visual_heading", name="Heading")
        # yaw error
        axis_error_yaw : PerceptionFieldAxis = PerceptionFieldAxis(type="angle", data_label="error_yaw", name="Yaw Error")


        # 2D xy grid
        error_field, _ = analyzer.analyzeXY(axis_x, axis_y)
        visualize(error_field, prefix="XY", save_dir=result_root_directory)

        # distance-visual_heading grid
        error_field_dist_heading, _ = analyzer.analyzeXY(axis_dist, axis_heding)
        visualize(error_field_dist_heading, prefix="dist_heading", save_dir=result_root_directory)


        # Dist-error grid
        prefix : str = "dist_delta-error"
        error_field_range, _ = analyzer.analyzeXY(axis_dist, axis_error_delta)
        field = error_field_range
        numb = field.num
        numb[numb == 0] = np.nan
        numb_log:np.ndarray = np.log10(field.num)
        # plot
        figures = []
        figures.append(PerceptionFieldPlot(prefix + "_" + "numb_log"))
        figures[-1].pcolormesh(field.mesh_center_x, field.mesh_center_y, numb_log, vmin=0)
        figures[-1].setAxes(field)


        # heading-yaw_error grid
        prefix = "yaw_error"
        error_field_yaw_error, _ = analyzer.analyzeXY(axis_heding, axis_error_yaw)
        field = error_field_yaw_error
        numb = field.num
        numb[numb == 0] = np.nan
        numb_log:np.ndarray = np.log10(field.num)
        # plot
        figures.append(PerceptionFieldPlot(prefix + "_" + "numb_log"))
        figures[-1].pcolormesh(field.mesh_center_x * 180.0 / np.pi, field.mesh_center_y * 180.0 / np.pi, numb_log, vmin=0)
        figures[-1].setAxes(field)


        # Save plots
        for fig in figures:
            fig.figure.savefig(Path(result_root_directory, "plot", fig.name + ".png"))


        # for debug
        # print(analyzer.df)
        # print(analyzer.df.columns)


        # Show plots
        plt.show()




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
