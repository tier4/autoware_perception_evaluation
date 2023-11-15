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
        self.setXY()

    def contourf(self, x, y, z, **kwargs):
        self.cs = self.ax.contourf(x, y, z, **kwargs)
        self.ax.contour(self.cs, colors='k')
        self.cbar = self.figure.colorbar(self.cs)
        self.setXY()

    def setXY(self)->None:
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_aspect('equal')
        self.ax.set_xlim([-110, 110])
        self.ax.set_ylim([-110, 110])
        self.ax.grid(c='k', ls='-', alpha=0.3)

def visualize(field:PerceptionAnalyzer3DField, result_root_directory:str)->None:
    
    # Plot
    figures = []
    figures.append(PerceptionFieldPlot("num_pair"))
    figures[-1].pcolormesh(field.mesh_center_x, field.mesh_center_y, field.num_pair, vmin=0)

    figures.append(PerceptionFieldPlot("ratio_tp"))
    figures[-1].pcolormesh(field.mesh_center_x, field.mesh_center_y, field.ratio_tp, vmin=0, vmax=1)

    figures.append(PerceptionFieldPlot("ratio_fp"))
    figures[-1].pcolormesh(field.mesh_center_x, field.mesh_center_y, field.ratio_fp, vmin=0, vmax=1)

    # figures.append(PerceptionFieldPlot("delta_mean"))
    # figures[-1].ax.matshow(field.error_delta_mean, cmap='jet')

    # figures.append(PerceptionFieldPlot("delta_mean_contour"))
    # levels = [0,0.1,0.2,0.3,0.5,1,2,5,10]
    # figures[-1].contourf(field.x, field.y, field.error_delta_mean, 
    #                      levels=levels, cmap='jet', corner_mask=False)

    figures.append(PerceptionFieldPlot("delta_mean_mesh"))
    figures[-1].pcolormesh(field.x, field.y, field.error_delta_mean, 
                           vmin=0, vmax = np.nanmax(field.error_delta_mean))
    cs = figures[-1].ax.scatter(field.x, field.y, marker='+',c='r', s=10)

    # Save result
    plot_file_path = Path(result_root_directory, "plot")
        
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

        # Analyze
        # conditions: object class, grid type, error/uncertainty...

        # Defalult analysis : all object class, all grid type, error
        # error_field, uncertainty_field = analyzer.analyze()

        # Specific class
        # labels: car, truck, bicycle, pedestrian, motorbike
        # src/simulator/perception_eval/docs/en/perception/label.md
        # error_field, _ = analyzer.analyze(label="car") 


        # Specific grid type - 2D xy grid
        # Initialize fields
        grid_axis_x : np.ndarray = np.array([-90, -60, -40, -30, -20, -10, -5, 5, 10, 20, 30, 40, 60, 90])
        grid_axis_y : np.ndarray = np.array([-90, -60, -40, -30, -20, -10, -5, 5, 10, 20, 30, 40, 60, 90])

        # Set condition
        error_field, uncertainty_field = analyzer.analyzeXY("x",grid_axis_x,"y",grid_axis_y)

        # Specific grid type - 1D range grid





        # print(analyzer.df)
        print(analyzer.df.columns)

        # Plot
        visualize(error_field, result_root_directory)

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
