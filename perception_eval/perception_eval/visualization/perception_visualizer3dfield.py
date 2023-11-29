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

# cspell: ignore figname, cbar, valuemap, yerr

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from perception_eval.tool import PerceptionFieldXY


class PerceptionFieldPlot:
    def __init__(self, figname: str, value: str = "Value []") -> None:
        self.name: str = figname
        self.figure = plt.figure(self.name, figsize=(10, 8))
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect("equal")
        self.value: str = value

    def contourf(self, x, y, z, **kwargs):
        self.cs = self.ax.contourf(x, y, z, **kwargs)
        self.ax.contour(self.cs, colors="k")
        self.cbar = self.figure.colorbar(self.cs)
        self.cbar.set_label(self.value)

    def set_axis_1d(self, field: PerceptionFieldXY, value: np.ndarray) -> None:
        if np.all(np.isnan(value)):
            return
        value_range = [np.nanmin(value), np.nanmax(value)]
        if value_range[0] == value_range[1]:
            value_range[0] -= 1
            value_range[1] += 1

        self.ax.set_xlabel(field.axis_x.get_title())
        self.ax.set_ylabel(self.value)

        value_scale = (value_range[1] - value_range[0]) / 10.0
        value_plot_margin = value_scale / 2.0  # 5% margin
        self.ax.set_aspect(field.axis_x.plot_aspect_ratio / value_scale)
        self.ax.set_xlim(field.axis_x.plot_range)
        self.ax.set_ylim(value_range[0] - value_plot_margin, value_range[1] + value_plot_margin)
        self.ax.grid(c="k", ls="-", alpha=0.3)
        self.ax.set_xticks(field.axis_x.grid_axis * field.axis_x.plot_scale)

    def set_axes(self, field: PerceptionFieldXY) -> None:
        self.ax.set_xlabel(field.axis_x.get_title())
        self.ax.set_ylabel(field.axis_y.get_title())
        self.ax.set_aspect(field.axis_x.plot_aspect_ratio / field.axis_y.plot_aspect_ratio)
        self.ax.set_xlim(field.axis_x.plot_range)
        self.ax.set_ylim(field.axis_y.plot_range)
        self.ax.grid(c="k", ls="-", alpha=0.3)
        self.ax.set_xticks(field.axis_x.grid_axis * field.axis_x.plot_scale)
        self.ax.set_yticks(field.axis_y.grid_axis * field.axis_y.plot_scale)

    def plot_mesh_map(self, field: PerceptionFieldXY, valuemap: np.ndarray, **kwargs) -> None:
        x: np.ndarray = field.mesh_x * field.axis_x.plot_scale
        y: np.ndarray = field.mesh_y * field.axis_y.plot_scale
        self.cs = self.ax.pcolormesh(x, y, valuemap, **kwargs)
        self.cbar = self.figure.colorbar(self.cs)
        self.cbar.set_label(self.value)

    def plot_scatter(self, x, y, **kwargs) -> None:
        self.cs = self.ax.scatter(x, y, **kwargs)

    def plot_scatter_3d(self, x, y, z, **kwargs) -> None:
        self.ax.remove()
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.cs = self.ax.scatter(x, y, z, **kwargs)
        # enlarge figure size
        self.figure.set_size_inches(14, 10)
        self.ax.set_zlabel(self.value)


class PerceptionFieldPlots:
    def __init__(self, save_dir: str) -> None:
        self.save_dir = save_dir
        self.figures: list[PerceptionFieldPlot] = []

    def add(self, figure: PerceptionFieldPlot) -> None:
        self.figures.append(figure)

    def save(self) -> None:
        for fig in self.figures:
            fig.figure.savefig(Path(self.save_dir, fig.name + ".png"))

    def show(self) -> None:
        plt.show()

    def close(self) -> None:
        plt.close("all")

    @property
    def last(self) -> PerceptionFieldPlot:
        return self.figures[-1]

    def plot_field_basics(self, field: PerceptionFieldXY, prefix: str, is_uncertainty: bool = False) -> None:
        # Preprocess
        mask_layer: np.ndarray = np.zeros(np.shape(field.num_pair), dtype=np.bool_)
        mask_layer[field.num_pair == 0] = True
        field.num_pair[mask_layer] = np.nan

        if is_uncertainty:
            prefix = prefix + "_uncertainty"

        # Plot
        # Number of data
        self.add(PerceptionFieldPlot(prefix + "_" + "num", "Samples [-]"))
        self.last.plot_mesh_map(field, field.num, vmin=0)
        self.last.set_axes(field)

        # True positive rate
        self.add(PerceptionFieldPlot(prefix + "_" + "ratio_tp", "True Positive rate [-]"))
        self.last.plot_mesh_map(field, field.ratio_tp, vmin=0, vmax=1)
        self.last.set_axes(field)

        # False positive rate
        self.add(PerceptionFieldPlot(prefix + "_" + "ratio_fp", "False Positive rate [-]"))
        self.last.plot_mesh_map(field, field.ratio_fp, vmin=0, vmax=1)
        self.last.set_axes(field)

        if is_uncertainty == False:
            # False negative rate
            self.add(PerceptionFieldPlot(prefix + "_" + "ratio_fn", "False Negative rate [-]"))
            self.last.plot_mesh_map(field, field.ratio_fn, vmin=0, vmax=1)
            self.last.set_axes(field)

        # Position error
        if field.has_any_error_data:
            title: str = "Position uncertainty [m]" if is_uncertainty else "Position error [m]"
            self.add(PerceptionFieldPlot(prefix + "_" + "delta_mean_mesh", title))
            vmax = 1
            if bool(np.all(np.isnan(field.error_delta_mean))) is False:
                vmax = np.nanmax(field.error_delta_mean)
            self.last.plot_mesh_map(field, field.error_delta_mean, vmin=0, vmax=vmax)

            # mean positions of each grid
            if hasattr(field, field.axis_x.data_label):
                x_mean_plot = getattr(field, field.axis_x.data_label) * field.axis_x.plot_scale
            else:
                x_mean_plot = field.mesh_center_x * field.axis_x.plot_scale
            if hasattr(field, field.axis_y.data_label):
                y_mean_plot = getattr(field, field.axis_y.data_label) * field.axis_y.plot_scale
            else:
                y_mean_plot = field.mesh_center_y * field.axis_y.plot_scale

            _ = self.last.ax.scatter(x_mean_plot, y_mean_plot, marker="+", c="r", s=10)
            self.last.set_axes(field)
        else:
            print("Plot (Prefix " + prefix + "): No TP data, nothing for error analysis")

    def plot_custom_field(
        self, field: PerceptionFieldXY, array: np.ndarray, filename: str, title: str, **kwargs
    ) -> None:
        self.add(PerceptionFieldPlot(filename, title))
        self.last.plot_mesh_map(field, array, **kwargs)
        self.last.set_axes(field)

    def plot_axis_basic(self, field: PerceptionFieldXY, prefix: str, is_uncertainty: bool = False) -> None:
        if is_uncertainty:
            prefix = prefix + "_uncertainty"

        self.add(PerceptionFieldPlot(prefix + "_" + "numb", "Samples [-]"))
        self.last.ax.scatter(field.dist, field.num, marker="x", c="r", s=10, label="Samples")
        self.last.set_axis_1d(field, field.num)

        self.add(PerceptionFieldPlot(prefix + "_" + "rates", "TP/FN/FP Rates [-]"))
        self.last.ax.scatter(field.dist, field.ratio_tp, marker="o", c="b", s=20, label="TP")
        self.last.ax.scatter(field.dist, field.ratio_fn, marker="x", c="r", s=20, label="FN")
        self.last.ax.scatter(field.dist, field.ratio_fp, marker="^", c="g", s=20, label="FP")
        self.last.set_axis_1d(field, field.ratio_tp)
        self.last.ax.set_ylim([0, 1])
        self.last.ax.set_aspect(10.0 / 0.2)
        self.last.ax.legend()

        if field.has_any_error_data == True:
            title: str = "Position error [m]"
            if is_uncertainty:
                title = "Position uncertainty [m]"
            self.add(PerceptionFieldPlot(prefix + "_" + "error_delta_bar", title))
            self.last.ax.errorbar(
                field.dist.flatten(),
                field.error_delta_mean.flatten(),
                yerr=field.error_delta_std.flatten(),
                marker="x",
                c="r",
            )
            self.last.set_axis_1d(field, field.error_delta_mean)
            self.last.ax.set_aspect(10.0 / 1)
            self.last.ax.set_ylim([-1, 5])
        else:
            print("Plot (Prefix " + prefix + "): No TP data, nothing for error analysis")
